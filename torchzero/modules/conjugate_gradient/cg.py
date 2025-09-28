from abc import ABC, abstractmethod
from typing import Literal

import torch

from ...core import Chainable, TensorTransform

from ...utils import TensorList, safe_dict_update_, unpack_dicts, unpack_states
from ..quasi_newton.quasi_newton import HessianUpdateStrategy
from ..opt_utils import safe_clip


class ConguateGradientBase(TensorTransform, ABC):
    """Base class for conjugate gradient methods. The only difference between them is how beta is calculated.

    This is an abstract class, to use it, subclass it and override `get_beta`.


    Args:
        defaults (dict | None, optional): dictionary of settings defaults. Defaults to None.
        clip_beta (bool, optional): whether to clip beta to be no less than 0. Defaults to False.
        restart_interval (int | None | Literal["auto"], optional):
            interval between resetting the search direction.
            "auto" means number of dimensions + 1, None means no reset. Defaults to None.
        inner (Chainable | None, optional): previous direction is added to the output of this module. Defaults to None.

    Example:

    ```python
    class PolakRibiere(ConguateGradientBase):
        def __init__(
            self,
            clip_beta=True,
            restart_interval: int | None = None,
            inner: Chainable | None = None
        ):
            super().__init__(clip_beta=clip_beta, restart_interval=restart_interval, inner=inner)

        def get_beta(self, p, g, prev_g, prev_d):
            denom = prev_g.dot(prev_g)
            if denom.abs() <= torch.finfo(g[0].dtype).eps: return 0
            return g.dot(g - prev_g) / denom
    ```

    """
    def __init__(self, defaults, clip_beta: bool, restart_interval: int | None | Literal['auto'], inner: Chainable | None = None):
        if defaults is None: defaults = {}
        safe_dict_update_(defaults, dict(restart_interval=restart_interval, clip_beta=clip_beta))
        super().__init__(defaults, inner=inner)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_grad')
        self.global_state.pop('stage', None)
        self.global_state['step'] = self.global_state.get('step', 1) - 1

    def initialize(self, p: TensorList, g: TensorList):
        """runs on first step when prev_grads and prev_dir are not available"""

    @abstractmethod
    def get_beta(self, p: TensorList, g: TensorList, prev_g: TensorList, prev_d: TensorList) -> float | torch.Tensor:
        """returns beta"""

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        params = TensorList(params)
        self.increment_counter("step", start=0)

        # initialize on first step
        if self.global_state.get('stage', "first update") == "first update":
            g_prev, d_prev = unpack_states(states, tensors, 'g_prev', 'd_prev', cls=TensorList)
            d_prev.copy_(tensors)
            g_prev.copy_(tensors)
            self.initialize(params, tensors)
            self.global_state['stage'] = "first apply"

        else:
            # if `update_tensors` was called multiple times before `apply_tensors`,
            # stage becomes 2
            self.global_state['stage'] = "initialized"

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        step = self.global_state['step']

        assert self.global_state['stage'] != "first update"

        # on 1st apply we don't have previous gradients
        # so just return tensors
        if self.global_state['stage'] == "first apply":
            self.global_state['stage'] = "initialized"
            return tensors

        params = TensorList(params)
        g_prev, d_prev = unpack_states(states, tensors, 'g_prev', 'd_prev', cls=TensorList)

        # get beta
        beta = self.get_beta(params, tensors, g_prev, d_prev)
        if settings[0]['clip_beta']: beta = max(0, beta) # pyright:ignore[reportArgumentType]

        # inner step
        # calculate new direction with beta
        dir = tensors.add_(d_prev.mul_(beta))
        d_prev.copy_(dir)

        # resetting every `reset_interval` steps, use step+1 to not reset on 1st step
        # so if reset_interval=2, then 1st step collects g_prev and d_prev, then
        # two steps will happen until reset.
        restart_interval = settings[0]['restart_interval']
        if restart_interval == 'auto': restart_interval = tensors.global_numel() + 1

        if restart_interval is not None and (step + 1) % restart_interval == 0:
            self.state.clear()
            self.global_state.clear()

        return dir

# ------------------------------- Polak-Ribière ------------------------------ #
def polak_ribiere_beta(g: TensorList, prev_g: TensorList):
    denom = prev_g.dot(prev_g)
    if denom.abs() <= torch.finfo(g[0].dtype).tiny * 2: return 0
    return g.dot(g - prev_g) / denom

class PolakRibiere(ConguateGradientBase):
    """Polak-Ribière-Polyak nonlinear conjugate gradient method.

    Note:
        This requires step size to be determined via a line search, so put a line search like ``tz.m.StrongWolfe(c2=0.1, a_init="first-order")`` after this.
    """
    def __init__(self, clip_beta=True, restart_interval: int | None | Literal['auto'] = 'auto', inner: Chainable | None = None):
        super().__init__({}, clip_beta=clip_beta, restart_interval=restart_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return polak_ribiere_beta(g, prev_g)

# ------------------------------ Fletcher–Reeves ----------------------------- #
def fletcher_reeves_beta(gg: torch.Tensor, prev_gg: torch.Tensor):
    if prev_gg.abs() <= torch.finfo(gg.dtype).tiny * 2: return 0
    return gg / prev_gg

class FletcherReeves(ConguateGradientBase):
    """Fletcher–Reeves nonlinear conjugate gradient method.

    Note:
        This requires step size to be determined via a line search, so put a line search like ``tz.m.StrongWolfe(c2=0.1, a_init="first-order")`` after this.
    """
    def __init__(self, restart_interval: int | None | Literal['auto'] = 'auto', clip_beta=False, inner: Chainable | None = None):
        super().__init__({}, clip_beta=clip_beta, restart_interval=restart_interval, inner=inner)

    def initialize(self, p, g):
        self.global_state['prev_gg'] = g.dot(g)

    def get_beta(self, p, g, prev_g, prev_d):
        gg = g.dot(g)
        beta = fletcher_reeves_beta(gg, self.global_state['prev_gg'])
        self.global_state['prev_gg'] = gg
        return beta

# ----------------------------- Hestenes–Stiefel ----------------------------- #
def hestenes_stiefel_beta(g: TensorList, prev_d: TensorList,prev_g: TensorList):
    grad_diff = g - prev_g
    denom = prev_d.dot(grad_diff)
    if denom.abs() < torch.finfo(g[0].dtype).tiny * 2: return 0
    return (g.dot(grad_diff) / denom).neg()


class HestenesStiefel(ConguateGradientBase):
    """Hestenes–Stiefel nonlinear conjugate gradient method.

    Note:
        This requires step size to be determined via a line search, so put a line search like ``tz.m.StrongWolfe(c2=0.1, a_init="first-order")`` after this.
    """
    def __init__(self, restart_interval: int | None | Literal['auto'] = 'auto', clip_beta=False, inner: Chainable | None = None):
        super().__init__({}, clip_beta=clip_beta, restart_interval=restart_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return hestenes_stiefel_beta(g, prev_d, prev_g)


# --------------------------------- Dai–Yuan --------------------------------- #
def dai_yuan_beta(g: TensorList, prev_d: TensorList,prev_g: TensorList):
    denom = prev_d.dot(g - prev_g)
    if denom.abs() <= torch.finfo(g[0].dtype).tiny * 2: return 0
    return (g.dot(g) / denom).neg()

class DaiYuan(ConguateGradientBase):
    """Dai–Yuan nonlinear conjugate gradient method.

    Note:
        This requires step size to be determined via a line search, so put a line search like ``tz.m.StrongWolfe(c2=0.1)`` after this.
    """
    def __init__(self, restart_interval: int | None | Literal['auto'] = 'auto', clip_beta=False, inner: Chainable | None = None):
        super().__init__({}, clip_beta=clip_beta, restart_interval=restart_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return dai_yuan_beta(g, prev_d, prev_g)


# -------------------------------- Liu-Storey -------------------------------- #
def liu_storey_beta(g:TensorList, prev_d:TensorList, prev_g:TensorList, ):
    denom = prev_g.dot(prev_d)
    if denom.abs() <= torch.finfo(g[0].dtype).tiny * 2: return 0
    return g.dot(g - prev_g) / denom

class LiuStorey(ConguateGradientBase):
    """Liu-Storey nonlinear conjugate gradient method.

    Note:
        This requires step size to be determined via a line search, so put a line search like ``tz.m.StrongWolfe(c2=0.1, a_init="first-order")`` after this.
    """
    def __init__(self, restart_interval: int | None | Literal['auto'] = 'auto', clip_beta=False, inner: Chainable | None = None):
        super().__init__({}, clip_beta=clip_beta, restart_interval=restart_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return liu_storey_beta(g, prev_d, prev_g)

# ----------------------------- Conjugate Descent ---------------------------- #
def conjugate_descent_beta(g:TensorList, prev_d:TensorList, prev_g:TensorList):
    denom = prev_g.dot(prev_d)
    if denom.abs() <= torch.finfo(g[0].dtype).tiny * 2: return 0
    return g.dot(g) / denom

class ConjugateDescent(ConguateGradientBase):
    """Conjugate Descent (CD).

    Note:
        This requires step size to be determined via a line search, so put a line search like ``tz.m.StrongWolfe(c2=0.1, a_init="first-order")`` after this.
    """
    def __init__(self, restart_interval: int | None | Literal['auto'] = 'auto', clip_beta=False, inner: Chainable | None = None):
        super().__init__({}, clip_beta=clip_beta, restart_interval=restart_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return conjugate_descent_beta(g, prev_d, prev_g)


# -------------------------------- Hager-Zhang ------------------------------- #
def hager_zhang_beta(g:TensorList, prev_d:TensorList, prev_g:TensorList,):
    g_diff = g - prev_g
    denom = prev_d.dot(g_diff)
    if denom.abs() <= torch.finfo(g[0].dtype).tiny * 2: return 0

    term1 = 1/denom
    # term2
    term2 = (g_diff - (2 * prev_d * (g_diff.pow(2).global_sum()/denom))).dot(g)
    return (term1 * term2).neg()


class HagerZhang(ConguateGradientBase):
    """Hager-Zhang nonlinear conjugate gradient method,

    Note:
        This requires step size to be determined via a line search, so put a line search like ``tz.m.StrongWolfe(c2=0.1, a_init="first-order")`` after this.
    """
    def __init__(self, restart_interval: int | None | Literal['auto'] = 'auto', clip_beta=False, inner: Chainable | None = None):
        super().__init__({}, clip_beta=clip_beta, restart_interval=restart_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return hager_zhang_beta(g, prev_d, prev_g)


# ----------------------------------- DYHS ---------------------------------- #
def dyhs_beta(g: TensorList, prev_d: TensorList,prev_g: TensorList):
    grad_diff = g - prev_g
    denom = prev_d.dot(grad_diff)
    if denom.abs() <= torch.finfo(g[0].dtype).tiny * 2: return 0

    # Dai-Yuan
    dy_beta = (g.dot(g) / denom).neg().clamp(min=0)

    # Hestenes–Stiefel
    hs_beta = (g.dot(grad_diff) / denom).neg().clamp(min=0)

    return max(0, min(dy_beta, hs_beta)) # type:ignore

class DYHS(ConguateGradientBase):
    """Dai-Yuan - Hestenes–Stiefel hybrid conjugate gradient method.

    Note:
        This requires step size to be determined via a line search, so put a line search like ``tz.m.StrongWolfe(c2=0.1, a_init="first-order")`` after this.
    """
    def __init__(self, restart_interval: int | None | Literal['auto'] = 'auto', clip_beta=False, inner: Chainable | None = None):
        super().__init__({}, clip_beta=clip_beta, restart_interval=restart_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return dyhs_beta(g, prev_d, prev_g)


def projected_gradient_(H:torch.Tensor, y:torch.Tensor):
    Hy = H @ y
    yHy = safe_clip(y.dot(Hy))
    H -= (Hy.outer(y) @ H) / yHy
    return H

class ProjectedGradientMethod(HessianUpdateStrategy): # this doesn't maintain hessian
    """Projected gradient method. Directly projects the gradient onto subspace conjugate to past directions.

    Notes:
        - This method uses N^2 memory.
        - This requires step size to be determined via a line search, so put a line search like ``tz.m.StrongWolfe(c2=0.1, a_init="first-order")`` after this.
        - This is not the same as projected gradient descent.

    Reference:
        Pearson, J. D. (1969). Variable metric methods of minimisation. The Computer Journal, 12(2), 171–178. doi:10.1093/comjnl/12.2.171.  (algorithm 5 in section 6)

    """

    def __init__(
        self,
        init_scale: float | Literal["auto"] = 1,
        tol: float = 1e-32,
        ptol: float | None = 1e-32,
        ptol_restart: bool = False,
        gtol: float | None = 1e-32,
        restart_interval: int | None | Literal['auto'] = 'auto',
        beta: float | None = None,
        update_freq: int = 1,
        scale_first: bool = False,
        concat_params: bool = True,
        # inverse: bool = True,
        inner: Chainable | None = None,
    ):
        super().__init__(
            defaults=None,
            init_scale=init_scale,
            tol=tol,
            ptol=ptol,
            ptol_restart=ptol_restart,
            gtol=gtol,
            restart_interval=restart_interval,
            beta=beta,
            update_freq=update_freq,
            scale_first=scale_first,
            concat_params=concat_params,
            inverse=True,
            inner=inner,
        )



    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, setting):
        return projected_gradient_(H=H, y=y)

