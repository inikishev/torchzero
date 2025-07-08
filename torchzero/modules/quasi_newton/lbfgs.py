from collections import deque
from operator import itemgetter

import torch

from ...core import Chainable, Module, Transform, Var, apply_transform
from ...utils import NumberList, TensorList, as_tensorlist, unpack_dicts, unpack_states
from ..functional import safe_scaling_


def _adaptive_damping(
    s: TensorList,
    y: TensorList,
    sy: torch.Tensor,
    init_damping = 0.99,
    eigval_bounds = (0.01, 1.5)
):
    # adaptive damping Al-Baali, M.: Quasi-Wolfe conditions for quasi-Newton methods for large-scale optimization. In: 40th Workshop on Large Scale Nonlinear Optimization, Erice, Italy, June 22–July 1 (2004)
    sigma_l, sigma_h = eigval_bounds
    u = sy / s.dot(s)
    if u <= sigma_l < 1: tau = min((1-sigma_l)/(1-u), init_damping)
    elif u >= sigma_h > 1: tau = min((sigma_h-1)/(u-1), init_damping)
    else: tau = init_damping
    y = tau * y + (1-tau) * s
    sy = s.dot(y)

    return s, y, sy

def lbfgs(
    tensors_: TensorList,
    s_history: deque[TensorList],
    y_history: deque[TensorList],
    sy_history: deque[torch.Tensor],
    y: TensorList | None,
    sy: torch.Tensor | None,
    z_beta: float | None,
    z_ema: TensorList | None,
    step: int,
):
    if len(s_history) == 0 or y is None or sy is None:

        # initial step size guess modified from pytorch L-BFGS
        return safe_scaling_(TensorList(tensors_))

    # 1st loop
    alpha_list = []
    q = tensors_.clone()
    for s_i, y_i, sy_i in zip(reversed(s_history), reversed(y_history), reversed(sy_history)):
        p_i = 1 / sy_i # this is also denoted as ρ (rho)
        alpha = p_i * s_i.dot(q)
        alpha_list.append(alpha)
        q.sub_(y_i, alpha=alpha) # pyright: ignore[reportArgumentType]

    # calculate z
    # s.y/y.y is also this weird y-looking symbol I couldn't find
    # z is it times q
    # actually H0 = (s.y/y.y) * I, and z = H0 @ q
    z = q * (sy / (y.dot(y)))

    # an attempt into adding momentum, lerping initial z seems stable compared to other variables
    if z_beta is not None:
        assert z_ema is not None
        if step == 1: z_ema.copy_(z)
        else: z_ema.lerp(z, 1-z_beta)
        z = z_ema

    # 2nd loop
    for s_i, y_i, sy_i, alpha_i in zip(s_history, y_history, sy_history, reversed(alpha_list)):
        p_i = 1 / sy_i
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

class LBFGS(Transform):
    """Limited-memory BFGS algorithm. A line search is recommended, although L-BFGS may be reasonably stable without it.

    Args:
        history_size (int, optional):
            number of past parameter differences and gradient differences to store. Defaults to 10.
        damping (bool, optional):
            whether to use adaptive damping. Learning rate might need to be lowered with this enabled. Defaults to False.
        init_damping (float, optional):
            initial damping for adaptive dampening. Defaults to 0.9.
        eigval_bounds (tuple, optional):
            eigenvalue bounds for adaptive dampening. Defaults to (0.5, 50).
        tol (float | None, optional):
            tolerance for minimal parameter difference to avoid instability. Defaults to 1e-10.
        tol_reset (bool, optional):
            If true, whenever gradient difference is less then `tol`, the history will be reset. Defaults to None.
        gtol (float | None, optional):
            tolerance for minimal gradient difference to avoid instability when there is no curvature. Defaults to 1e-10.
        params_beta (float | None, optional):
            if not None, EMA of parameters is used for preconditioner update. Defaults to None.
        grads_beta (float | None, optional):
            if not None, EMA of gradients is used for preconditioner update. Defaults to None.
        update_freq (int, optional):
            how often to update L-BFGS history. Defaults to 1.
        z_beta (float | None, optional):
            optional EMA for initial H^-1 @ q. Acts as a kind of momentum but is prone to get stuck. Defaults to None.
        inner (Chainable | None, optional):
            optional inner modules applied after updating L-BFGS history and before preconditioning. Defaults to None.

    Examples:
        L-BFGS with strong-wolfe line search

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.LBFGS(100),
                tz.m.StrongWolfe()
            )

        Dampened L-BFGS

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.LBFGS(damping=True),
                tz.m.StrongWolfe()
            )

        L-BFGS preconditioning applied to momentum (may be unstable!)

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.LBFGS(inner=tz.m.EMA(0.9)),
                tz.m.LR(1e-2)
            )
    """
    def __init__(
        self,
        history_size=10,
        damping: bool = False,
        init_damping=0.9,
        eigval_bounds=(0.5, 50),
        tol: float | None = 1e-10,
        tol_reset: bool = False,
        gtol: float | None = 1e-10,
        params_beta: float | None = None,
        grads_beta: float | None = None,
        update_freq = 1,
        z_beta: float | None = None,
        inner: Chainable | None = None,
    ):
        defaults = dict(history_size=history_size, tol=tol, gtol=gtol, damping=damping, init_damping=init_damping, eigval_bounds=eigval_bounds, params_beta=params_beta, grads_beta=grads_beta, update_freq=update_freq, z_beta=z_beta, tol_reset=tol_reset)
        super().__init__(defaults, uses_grad=False, inner=inner)

        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)
        self.global_state['sy_history'] = deque(maxlen=history_size)

    def reset(self):
        self.state.clear()
        self.global_state['step'] = 0
        self.global_state['s_history'].clear()
        self.global_state['y_history'].clear()
        self.global_state['sy_history'].clear()

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_l_params', 'prev_l_grad')
        self.global_state.pop('step', None)

    @torch.no_grad
    def update_tensors(self, tensors, params, grads, loss, states, settings):
        params = as_tensorlist(params)
        update = as_tensorlist(tensors)
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        # history of s and k
        s_history: deque[TensorList] = self.global_state['s_history']
        y_history: deque[TensorList] = self.global_state['y_history']
        sy_history: deque[torch.Tensor] = self.global_state['sy_history']

        damping,init_damping,eigval_bounds,update_freq = itemgetter('damping','init_damping','eigval_bounds','update_freq')(settings[0])
        params_beta, grads_beta = unpack_dicts(settings, 'params_beta', 'grads_beta')

        l_params, l_update = _lerp_params_update_(self, params, update, params_beta, grads_beta)
        prev_l_params, prev_l_grad = unpack_states(states, tensors, 'prev_l_params', 'prev_l_grad', cls=TensorList)

        # 1st step - there are no previous params and grads, lbfgs will do normalized SGD step
        if step == 0:
            s = None; y = None; sy = None
        else:
            s = l_params - prev_l_params
            y = l_update - prev_l_grad
            sy = s.dot(y)

            if damping:
                s, y, sy = _adaptive_damping(s, y, sy, init_damping=init_damping, eigval_bounds=eigval_bounds)

        prev_l_params.copy_(l_params)
        prev_l_grad.copy_(l_update)

        # update effective preconditioning state
        if step % update_freq == 0:
            if sy is not None and sy > 1e-10:
                assert s is not None and y is not None
                s_history.append(s)
                y_history.append(y)
                sy_history.append(sy)

        # store for apply
        self.global_state['s'] = s
        self.global_state['y'] = y
        self.global_state['sy'] = sy

    def make_Hv(self):
        ...

    def make_Bv(self):
        ...

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        tensors = as_tensorlist(tensors)

        s = self.global_state.pop('s')
        y = self.global_state.pop('y')
        sy = self.global_state.pop('sy')

        setting = settings[0]
        tol = setting['tol']
        gtol = setting['gtol']
        tol_reset = setting['tol_reset']
        z_beta = setting['z_beta']

        # tolerance on parameter difference to avoid exploding after converging
        if tol is not None:
            if s is not None and s.abs().global_max() <= tol:
                if tol_reset: self.reset()
                return safe_scaling_(TensorList(tensors))

        # tolerance on gradient difference to avoid exploding when there is no curvature
        if tol is not None:
            if y is not None and y.abs().global_max() <= gtol:
                return safe_scaling_(TensorList(tensors))

        # lerp initial H^-1 @ q guess
        z_ema = None
        if z_beta is not None:
            z_ema = unpack_states(states, tensors, 'z_ema', cls=TensorList)

        # precondition
        dir = lbfgs(
            tensors_=tensors,
            s_history=self.global_state['s_history'],
            y_history=self.global_state['y_history'],
            sy_history=self.global_state['sy_history'],
            y=y,
            sy=sy,
            z_beta = z_beta,
            z_ema = z_ema,
            step=self.global_state.get('step', 1)
        )

        return dir