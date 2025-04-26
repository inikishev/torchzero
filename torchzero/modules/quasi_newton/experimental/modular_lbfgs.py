from collections import deque
from operator import itemgetter
import torch

from ....core import Transform, Chainable, maybe_chain, Module, Vars, AnyTransform, apply_transform
from ....utils import TensorList, as_tensorlist, NumberList


def _adaptive_damping(
    s_k: TensorList,
    y_k: TensorList,
    ys_k: torch.Tensor,
    init_damping = 0.99,
    eigval_bounds = (0.01, 1.5)
):
    # adaptive damping Al-Baali, M.: Quasi-Wolfe conditions for quasi-Newton methods for large-scale optimization. In: 40th Workshop on Large Scale Nonlinear Optimization, Erice, Italy, June 22–July 1 (2004)
    sigma_l, sigma_h = eigval_bounds
    u = ys_k / s_k.dot(s_k)
    if u <= sigma_l < 1: tau = min((1-sigma_l)/(1-u), init_damping)
    elif u >= sigma_h > 1: tau = min((sigma_h-1)/(u-1), init_damping)
    else: tau = init_damping
    y_k = tau * y_k + (1-tau) * s_k
    ys_k = s_k.dot(y_k)

    return s_k, y_k, ys_k

def lbfgs(
    tensors_: TensorList,
    s_history: deque[TensorList],
    y_history: deque[TensorList],
    sy_history: deque[torch.Tensor],
    y_k: TensorList | None,
    ys_k: torch.Tensor | None,
    z_beta: float | None,
    z_ema: TensorList | None,
    step: int,
):
    if len(s_history) == 0 or y_k is None or ys_k is None:
        # dir = params.grad.sign() # may work fine

        # initial step size guess taken from pytorch L-BFGS
        return tensors_.mul_(min(1.0, 1.0 / tensors_.abs().global_sum())) # pyright: ignore[reportArgumentType]

    else:
        # 1st loop
        alpha_list = []
        q = tensors_.clone()
        for s_i, y_i, ys_i in zip(reversed(s_history), reversed(y_history), reversed(sy_history)):
            p_i = 1 / ys_i # this is also denoted as ρ (rho)
            alpha = p_i * s_i.dot(q)
            alpha_list.append(alpha)
            q.sub_(y_i, alpha=alpha) # pyright: ignore[reportArgumentType]

        # calculate z
        # s.y/y.y is also this weird y-looking symbol I couldn't find
        # z is it times q
        # actually H0 = (s.y/y.y) * I, and z = H0 @ q
        z = q * (ys_k / (y_k.dot(y_k)))

        # an attempt into adding momentum, lerping initial z seems stable compared to other variables
        if z_beta is not None:
            assert z_ema is not None
            if step == 0: z_ema.copy_(z)
            else: z_ema.lerp(z, 1-z_beta)
            z = z_ema

        # 2nd loop
        for s_i, y_i, ys_i, alpha_i in zip(s_history, y_history, sy_history, reversed(alpha_list)):
            p_i = 1 / ys_i
            beta_i = p_i * y_i.dot(z)
            z.add_(s_i, alpha = alpha_i - beta_i)

        return z

def _lerp_params_update_(
    self_: Module,
    params: list[torch.Tensor],
    update: list[torch.Tensor],
    params_beta: list[float | None],
    grads_beta: list[float | None],
):
    for i, (p, u, p_beta, u_beta) in enumerate(zip(params.copy(), update.copy(), params_beta, grads_beta)):
        if p_beta is not None or u_beta is not None:
            state = self_.state[p]

            if p_beta is not None:
                if 'param_ema' not in state: state['param_ema'] = p.clone()
                else: state['param_ema'].lerp_(p, 1-p_beta)
                params[i] = state['param_ema']

            if u_beta is not None:
                if 'grad_ema' not in state: state['grad_ema'] = u.clone()
                else: state['grad_ema'].lerp_(u, 1-u_beta)
                update[i] = state['grad_ema']

    return TensorList(params), TensorList(update)

def _apply_tfms_into_history(
    self: Module,
    params: list[torch.Tensor],
    vars: Vars,
    update: list[torch.Tensor],
):
    if 'params_history_tfm' in self.children:
        params = apply_transform(self.children['params_history_tfm'], target=as_tensorlist(params).clone(), params=params, grad=vars.grad, vars=vars)

    if 'grad_history_tfm' in self.children:
        update = apply_transform(self.children['grad_history_tfm'], target=as_tensorlist(update).clone(), params=params, grad=vars.grad, vars=vars)

    return params, update

def _apply_tfms_into_precond(
    self: Module,
    params: list[torch.Tensor],
    vars: Vars,
    update: list[torch.Tensor],
):
    if 'params_precond_tfm' in self.children:
        params = apply_transform(self.children['params_precond_tfm'], target=as_tensorlist(params).clone(), params=params, grad=vars.grad, vars=vars)

    if 'grad_precond_tfm' in self.children:
        update = apply_transform(self.children['grad_precond_tfm'], target=update, params=params, grad=vars.grad, vars=vars)

    return params, update


class ModularLBFGS(Module):
    """L-BFGS with ability to apply transforms to many inner variables.

    Args:
        history_size (int, optional): number of past parameter differences and gradient differences to store. Defaults to 10.
        tol (float | None, optional):
            tolerance for minimal gradient difference to avoid instability after converging to minima. Defaults to 1e-10.
        damping (bool, optional):
            whether to use adaptive damping. Learning rate might need to be lowered with this enabled. Defaults to False.
        init_damping (float, optional):
            initial damping for adaptive dampening. Defaults to 0.9.
        eigval_bounds (tuple, optional):
            eigenvalue bounds for adaptive dampening. Defaults to (0.5, 50).
        update_freq (int, optional):
            how often to update L-BFGS history. Defaults to 1.
        z_beta (float | None, optional):
            optional EMA for initial H^-1 @ q. Acts as a kind of momentum but is prone to get stuck. Defaults to None.
        params_history_tfm (AnyTransform | None, optional):
            transform module applied to params before adding s_k to history. Defaults to None.
        grad_history_tfm (AnyTransform | None, optional):
            transform module applied to grads before adding y_k to history. Defaults to None.
        params_precond_tfm (AnyTransform | None, optional):
            transform module applied to params to calculate s_k before preconditioning. Defaults to None.
        grad_precond_tfm (AnyTransform | None, optional):
            transform module applied to grads to calculate y_k before preconditioning. Defaults to None.
        update_precond_tfm (Chainable | None, optional):
            transform module applied to grads that are being preconditioned. Defaults to None.
    """
    def __init__(
        self,
        history_size=10,
        tol: float | None = 1e-10,
        damping: bool = False,
        init_damping=0.9,
        eigval_bounds=(0.5, 50),
        update_freq = 1,
        z_beta: float | None = None,
        params_history_tfm: AnyTransform | None = None,
        grad_history_tfm: AnyTransform | None = None,
        params_precond_tfm: AnyTransform | None = None,
        grad_precond_tfm: AnyTransform | None = None,
        update_precond_tfm: Chainable | None = None,
    ):
        defaults = dict(history_size=history_size, tol=tol, damping=damping, init_damping=init_damping, eigval_bounds=eigval_bounds, update_freq=update_freq, z_beta=z_beta)
        super().__init__(defaults)

        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)
        self.global_state['sy_history'] = deque(maxlen=history_size)
        self.global_state['step'] = 0

        for k,v in (('update_precond_tfm', update_precond_tfm), ('params_history_tfm', params_history_tfm), ('grad_history_tfm', grad_history_tfm),
                    ('params_precond_tfm', params_precond_tfm), ('grad_precond_tfm', grad_precond_tfm)):
            if v is not None:
                self.set_child(k,v)



    @torch.no_grad
    def step(self, vars):
        params = as_tensorlist(vars.params)
        update = as_tensorlist(vars.get_update())
        step = self.global_state['step']

        # history of s and k
        s_history: deque[TensorList] = self.global_state['s_history']
        y_history: deque[TensorList] = self.global_state['y_history']
        sy_history: deque[torch.Tensor] = self.global_state['sy_history']

        tol, damping, init_damping, eigval_bounds, update_freq, z_beta = itemgetter(
            'tol', 'damping', 'init_damping', 'eigval_bounds', 'update_freq', 'z_beta')(self.settings[params[0]])

        # params_beta, grads_beta = self.get_settings('params_beta', 'grads_beta', params=params, cls=NumberList)
        # l_params, l_update = _lerp_params_update_(self, params, update, params_beta, grads_beta)

        # params and update that go into history
        params_h, update_h = _apply_tfms_into_history(
            self,
            params=params,
            vars=vars,
            update=update,
        )

        prev_params_h, prev_grad_h = self.get_state('prev_params_h', 'prev_grad_h', params=params, cls=TensorList)

        # 1st step - there are no previous params and grads, `lbfgs` will do normalized SGD step
        if step == 0:
            s_k_h = None; y_k_h = None; ys_k_h = None
        else:
            s_k_h = params_h - prev_params_h
            y_k_h = update_h - prev_grad_h
            ys_k_h = s_k_h.dot(y_k_h)

            if damping:
                s_k_h, y_k_h, ys_k_h = _adaptive_damping(s_k_h, y_k_h, ys_k_h, init_damping=init_damping, eigval_bounds=eigval_bounds)

        prev_params_h.copy_(params_h)
        prev_grad_h.copy_(update_h)

        # update effective preconditioning state
        if step % update_freq == 0:
            if ys_k_h is not None and ys_k_h > 1e-10:
                assert s_k_h is not None and y_k_h is not None
                s_history.append(s_k_h)
                y_history.append(y_k_h)
                sy_history.append(ys_k_h)

        # step with inner module before applying preconditioner
        if 'update_precond_tfm' in self.children:
            update_precond_tfm = self.children['update_precond_tfm']
            inner_vars = update_precond_tfm.step(vars.clone(clone_update=True))
            vars.update_attrs_from_clone_(inner_vars)
            tensors = inner_vars.update
            assert tensors is not None
        else:
            tensors = update.clone()

        # lerp initial H^-1 @ q guess
        z_ema = None
        if z_beta is not None:
            z_ema = self.get_state('z_ema', params=vars.params, cls=TensorList)

        # transforms into preconditioner
        params_p, update_p = _apply_tfms_into_precond(self, params=params, vars=vars, update=update)
        prev_params_p, prev_grad_p = self.get_state('prev_params_p', 'prev_grad_p', params=params, cls=TensorList)

        if step == 0:
            s_k_p = None; y_k_p = None; ys_k_p = None

        else:
            s_k_p = params_p - prev_params_p
            y_k_p = update_p - prev_grad_p
            ys_k_p = s_k_p.dot(y_k_p)

            if damping:
                s_k_p, y_k_p, ys_k_p = _adaptive_damping(s_k_p, y_k_p, ys_k_p, init_damping=init_damping, eigval_bounds=eigval_bounds)

        prev_params_p.copy_(params_p)
        prev_grad_p.copy_(update_p)

        # tolerance on gradient difference to avoid exploding after converging
        if tol is not None:
            if y_k_p is not None and y_k_p.abs().global_max() <= tol:
                vars.update = update # may have been updated by inner module, probably makes sense to use it here?
                return vars

        # precondition
        dir = lbfgs(
            tensors_=as_tensorlist(tensors),
            s_history=s_history,
            y_history=y_history,
            sy_history=sy_history,
            y_k=y_k_p,
            ys_k=ys_k_p,
            z_beta = z_beta,
            z_ema = z_ema,
            step=step
        )

        self.global_state['step'] += 1
        vars.update = dir

        return vars

