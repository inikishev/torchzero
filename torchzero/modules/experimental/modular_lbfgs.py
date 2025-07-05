from collections import deque
from operator import itemgetter
from typing import Any

import torch

from ...core import Chainable, Module, Transform, Var, apply_transform, maybe_chain
from ...utils import NumberList, TensorList, as_tensorlist


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
    var: Var,
    s_history: deque[TensorList],
    y_history: deque[TensorList],
    sy_history: deque[torch.Tensor],
    y_k: TensorList | None,
    ys_k: torch.Tensor | None,
    z_tfm: Any,
):
    if len(s_history) == 0 or y_k is None or ys_k is None:

        # initial step size guess modified from pytorch L-BFGS
        scale = 1 / tensors_.abs().global_sum()
        if scale < 1e-5: scale = 1 / tensors_.abs().mean()
        return tensors_.mul_(min(1.0, scale)) # pyright: ignore[reportArgumentType]

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

    if z_tfm is not None:
        z = TensorList(apply_transform(z_tfm, tensors=z, params=var.params, grads=var.grad, var=var))

    # 2nd loop
    for s_i, y_i, ys_i, alpha_i in zip(s_history, y_history, sy_history, reversed(alpha_list)):
        p_i = 1 / ys_i
        beta_i = p_i * y_i.dot(z)
        z.add_(s_i, alpha = alpha_i - beta_i)

    return z

def _apply_tfms_into_history(
    self: Module,
    params: list[torch.Tensor],
    var: Var,
    update: list[torch.Tensor],
):
    if 'params_history_tfm' in self.children:
        params = apply_transform(self.children['params_history_tfm'], tensors=as_tensorlist(params).clone(), params=params, grads=var.grad, var=var)

    if 'grad_history_tfm' in self.children:
        update = apply_transform(self.children['grad_history_tfm'], tensors=as_tensorlist(update).clone(), params=params, grads=var.grad, var=var)

    return params, update

def _apply_tfms_into_precond(
    self: Module,
    params: list[torch.Tensor],
    var: Var,
    update: list[torch.Tensor],
):
    if 'params_precond_tfm' in self.children:
        params = apply_transform(self.children['params_precond_tfm'], tensors=as_tensorlist(params).clone(), params=params, grads=var.grad, var=var)

    if 'grad_precond_tfm' in self.children:
        update = apply_transform(self.children['grad_precond_tfm'], tensors=update, params=params, grads=var.grad, var=var)

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
        z_tfm (float | None, optional):
            transform module applied to initial H^-1 @ q guess. Defaults to None.
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
        params_history_tfm: Chainable | None = None,
        grad_history_tfm: Chainable | None = None,
        params_precond_tfm: Chainable | None = None,
        grad_precond_tfm: Chainable | None = None,
        update_precond_tfm: Chainable | None = None,
        z_tfm: Chainable | None = None,
    ):
        defaults = dict(history_size=history_size, tol=tol, damping=damping, init_damping=init_damping, eigval_bounds=eigval_bounds, update_freq=update_freq)
        super().__init__(defaults)

        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)
        self.global_state['sy_history'] = deque(maxlen=history_size)

        loc = locals().copy()
        for k in ('update_precond_tfm', 'params_history_tfm', 'grad_history_tfm', 'params_precond_tfm', 'grad_precond_tfm','z_tfm'):
            v = loc[k]
            if v is not None:
                self.set_child(k,v)

    def reset(self):
        """Resets the internal state of the L-SR1 module."""
        # super().reset() # Clears self.state (per-parameter) if any, and "step"
        self.state.clear()
        self.global_state['step'] = 0
        self.global_state['s_history'].clear()
        self.global_state['y_history'].clear()
        self.global_state['sy_history'].clear()

    @torch.no_grad
    def step(self, var):
        params = as_tensorlist(var.params)
        update = as_tensorlist(var.get_update())
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        # history of s and k
        s_history: deque[TensorList] = self.global_state['s_history']
        y_history: deque[TensorList] = self.global_state['y_history']
        sy_history: deque[torch.Tensor] = self.global_state['sy_history']

        tol, damping, init_damping, eigval_bounds, update_freq = itemgetter(
            'tol', 'damping', 'init_damping', 'eigval_bounds', 'update_freq')(self.settings[params[0]])

        # params_beta, grads_beta = self.get_settings('params_beta', 'grads_beta', params=params, cls=NumberList)
        # l_params, l_update = _lerp_params_update_(self, params, update, params_beta, grads_beta)

        # params and update that go into history
        params_h, update_h = _apply_tfms_into_history(
            self,
            params=params,
            var=var,
            update=update,
        )

        prev_params_h, prev_grad_h = self.get_state(params, 'prev_params_h', 'prev_grad_h', cls=TensorList)

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
            inner_var = update_precond_tfm.step(var.clone(clone_update=True))
            var.update_attrs_from_clone_(inner_var)
            tensors = inner_var.update
            assert tensors is not None
        else:
            tensors = update.clone()

        # transforms into preconditioner
        params_p, update_p = _apply_tfms_into_precond(self, params=params, var=var, update=update)
        prev_params_p, prev_grad_p = self.get_state(params, 'prev_params_p', 'prev_grad_p', cls=TensorList)

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
                var.update = update # may have been updated by inner module, probably makes sense to use it here?
                return var

        # precondition
        dir = lbfgs(
            tensors_=as_tensorlist(tensors),
            var=var,
            s_history=s_history,
            y_history=y_history,
            sy_history=sy_history,
            y_k=y_k_p,
            ys_k=ys_k_p,
            z_tfm=self.children.get('z_tfm', None),
        )

        var.update = dir

        return var

