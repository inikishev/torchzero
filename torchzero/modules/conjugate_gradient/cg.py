from abc import ABC, abstractmethod
from typing import Literal

import torch

from ...core import (
    Chainable,
    Modular,
    Module,
    Transform,
    Var,
    apply_transform,
)
from ...utils import TensorList, as_tensorlist, unpack_dicts, unpack_states
from ..line_search import LineSearchBase
from ..quasi_newton.quasi_newton import HessianUpdateStrategy
from ..functional import safe_clip


class ConguateGradientBase(Transform, ABC):
    """Base class for conjugate gradient methods. The only difference between them is how beta is calculated.

    This is an abstract class, to use it, subclass it and override `get_beta`.


    Args:
        defaults (dict | None, optional): dictionary of settings defaults. Defaults to None.
        clip_beta (bool, optional): whether to clip beta to be no less than 0. Defaults to False.
        reset_interval (int | None | Literal["auto"], optional):
            interval between resetting the search direction.
            "auto" means number of dimensions + 1, None means no reset. Defaults to None.
        inner (Chainable | None, optional): previous direction is added to the output of this module. Defaults to None.

    Example:

        .. code-block:: python

            class PolakRibiere(ConguateGradientBase):
                def __init__(
                    self,
                    clip_beta=True,
                    reset_interval: int | None = None,
                    inner: Chainable | None = None
                ):
                    super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

                def get_beta(self, p, g, prev_g, prev_d):
                    denom = prev_g.dot(prev_g)
                    if denom.abs() <= torch.finfo(g[0].dtype).eps: return 0
                    return g.dot(g - prev_g) / denom

    """
    def __init__(self, defaults = None, clip_beta: bool = False, reset_interval: int | None | Literal['auto'] = None, inner: Chainable | None = None):
        if defaults is None: defaults = {}
        defaults['reset_interval'] = reset_interval
        defaults['clip_beta'] = clip_beta
        super().__init__(defaults, uses_grad=False)

        if inner is not None:
            self.set_child('inner', inner)


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
    def update_tensors(self, tensors, params, grads, loss, states, settings):
        tensors = as_tensorlist(tensors)
        params = as_tensorlist(params)

        step = self.global_state.get('step', 0) + 1
        self.global_state['step'] = step

        # initialize on first step
        if self.global_state.get('stage', 0) == 0:
            g_prev, d_prev = unpack_states(states, tensors, 'g_prev', 'd_prev', cls=TensorList)
            d_prev.copy_(tensors)
            g_prev.copy_(tensors)
            self.initialize(params, tensors)
            self.global_state['stage'] = 1

        else:
            # if `update_tensors` was called multiple times before `apply_tensors`,
            # stage becomes 2
            self.global_state['stage'] = 2

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        tensors = as_tensorlist(tensors)
        step = self.global_state['step']

        if 'inner' in self.children:
            tensors = as_tensorlist(apply_transform(self.children['inner'], tensors, params, grads))

        assert self.global_state['stage'] != 0
        if self.global_state['stage'] == 1:
            self.global_state['stage'] = 2
            return tensors

        params = as_tensorlist(params)
        g_prev, d_prev = unpack_states(states, tensors, 'g_prev', 'd_prev', cls=TensorList)

        # get beta
        beta = self.get_beta(params, tensors, g_prev, d_prev)
        if settings[0]['clip_beta']: beta = max(0, beta) # pyright:ignore[reportArgumentType]

        # inner step
        # calculate new direction with beta
        dir = tensors.add_(d_prev.mul_(beta))
        d_prev.copy_(dir)

        # resetting
        reset_interval = settings[0]['reset_interval']
        if reset_interval == 'auto': reset_interval = tensors.global_numel() + 1
        if reset_interval is not None and step % reset_interval == 0:
            self.reset()

        return dir

# ------------------------------- Polak-Ribière ------------------------------ #
def polak_ribiere_beta(g: TensorList, prev_g: TensorList):
    denom = prev_g.dot(prev_g)
    if denom.abs() <= torch.finfo(g[0].dtype).eps: return 0
    return g.dot(g - prev_g) / denom

class PolakRibiere(ConguateGradientBase):
    """Polak-Ribière-Polyak nonlinear conjugate gradient method.

    .. note::
        - This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe(c2=0.1)` after this.
    """
    def __init__(self, clip_beta=True, reset_interval: int | None = None, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return polak_ribiere_beta(g, prev_g)

# ------------------------------ Fletcher–Reeves ----------------------------- #
def fletcher_reeves_beta(gg: torch.Tensor, prev_gg: torch.Tensor):
    if prev_gg.abs() <= torch.finfo(gg.dtype).eps: return 0
    return gg / prev_gg

class FletcherReeves(ConguateGradientBase):
    """Fletcher–Reeves nonlinear conjugate gradient method.

    .. note::
        - This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe(c2=0.1)` after this.
    """
    def __init__(self, reset_interval: int | None | Literal['auto'] = 'auto', clip_beta=False, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

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
    if denom.abs() < torch.finfo(g[0].dtype).eps: return 0
    return (g.dot(grad_diff) / denom).neg()


class HestenesStiefel(ConguateGradientBase):
    """Hestenes–Stiefel nonlinear conjugate gradient method.

    .. note::
        - This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe(c2=0.1)` after this.
    """
    def __init__(self, reset_interval: int | None | Literal['auto'] = None, clip_beta=False, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return hestenes_stiefel_beta(g, prev_d, prev_g)


# --------------------------------- Dai–Yuan --------------------------------- #
def dai_yuan_beta(g: TensorList, prev_d: TensorList,prev_g: TensorList):
    denom = prev_d.dot(g - prev_g)
    if denom.abs() <= torch.finfo(g[0].dtype).eps: return 0
    return (g.dot(g) / denom).neg()

class DaiYuan(ConguateGradientBase):
    """Dai–Yuan nonlinear conjugate gradient method.

    .. note::
        - This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe(c2=0.1)` after this. Although Dai–Yuan formula provides an automatic step size scaling so it is technically possible to omit line search and instead use a small step size.
    """
    def __init__(self, reset_interval: int | None | Literal['auto'] = None, clip_beta=False, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return dai_yuan_beta(g, prev_d, prev_g)


# -------------------------------- Liu-Storey -------------------------------- #
def liu_storey_beta(g:TensorList, prev_d:TensorList, prev_g:TensorList, ):
    denom = prev_g.dot(prev_d)
    if denom.abs() <= torch.finfo(g[0].dtype).eps: return 0
    return g.dot(g - prev_g) / denom

class LiuStorey(ConguateGradientBase):
    """Liu-Storey nonlinear conjugate gradient method.

    .. note::
        - This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe(c2=0.1)` after this.
    """
    def __init__(self, reset_interval: int | None | Literal['auto'] = None, clip_beta=False, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return liu_storey_beta(g, prev_d, prev_g)

# ----------------------------- Conjugate Descent ---------------------------- #
def conjugate_descent_beta(g:TensorList, prev_d:TensorList, prev_g:TensorList):
    denom = prev_g.dot(prev_d)
    if denom.abs() <= torch.finfo(g[0].dtype).eps: return 0
    return g.dot(g) / denom

class ConjugateDescent(ConguateGradientBase):
    """Conjugate Descent (CD).

    .. note::
        - This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe(c2=0.1)` after this.
    """
    def __init__(self, reset_interval: int | None | Literal['auto'] = None, clip_beta=False, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return conjugate_descent_beta(g, prev_d, prev_g)


# -------------------------------- Hager-Zhang ------------------------------- #
def hager_zhang_beta(g:TensorList, prev_d:TensorList, prev_g:TensorList,):
    g_diff = g - prev_g
    denom = prev_d.dot(g_diff)
    if denom.abs() <= torch.finfo(g[0].dtype).eps: return 0

    term1 = 1/denom
    # term2
    term2 = (g_diff - (2 * prev_d * (g_diff.pow(2).global_sum()/denom))).dot(g)
    return (term1 * term2).neg()


class HagerZhang(ConguateGradientBase):
    """Hager-Zhang nonlinear conjugate gradient method,

    .. note::
        - This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe(c2=0.1)` after this.
    """
    def __init__(self, reset_interval: int | None | Literal['auto'] = None, clip_beta=False, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return hager_zhang_beta(g, prev_d, prev_g)


# ----------------------------------- DYHS ---------------------------------- #
def dyhs_beta(g: TensorList, prev_d: TensorList,prev_g: TensorList):
    grad_diff = g - prev_g
    denom = prev_d.dot(grad_diff)
    if denom.abs() <= torch.finfo(g[0].dtype).eps: return 0

    # Dai-Yuan
    dy_beta = (g.dot(g) / denom).neg().clamp(min=0)

    # Hestenes–Stiefel
    hs_beta = (g.dot(grad_diff) / denom).neg().clamp(min=0)

    return max(0, min(dy_beta, hs_beta)) # type:ignore

class DYHS(ConguateGradientBase):
    """Dai-Yuan - Hestenes–Stiefel hybrid conjugate gradient method.

    .. note::
        - This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe(c2=0.1)` after this.
    """
    def __init__(self, reset_interval: int | None | Literal['auto'] = None, clip_beta=False, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return dyhs_beta(g, prev_d, prev_g)


def projected_gradient_(H:torch.Tensor, y:torch.Tensor):
    Hy = H @ y
    yHy = safe_clip(y.dot(Hy))
    H -= (Hy.outer(y) @ H) / yHy
    return H

class ProjectedGradientMethod(HessianUpdateStrategy): # this doesn't maintain hessian
    """Projected gradient method.

    .. note::
        This method uses N^2 memory.

    .. note::
        This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe(c2=0.1)` after this.

    .. note::
        This is not the same as projected gradient descent.

    Reference:
        Pearson, J. D. (1969). Variable metric methods of minimisation. The Computer Journal, 12(2), 171–178. doi:10.1093/comjnl/12.2.171.

    """

    def __init__(
        self,
        init_scale: float | Literal["auto"] = 1,
        tol: float = 1e-8,
        ptol: float | None = 1e-10,
        ptol_reset: bool = False,
        gtol: float | None = 1e-10,
        reset_interval: int | None | Literal['auto'] = 'auto',
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
            ptol_reset=ptol_reset,
            gtol=gtol,
            reset_interval=reset_interval,
            beta=beta,
            update_freq=update_freq,
            scale_first=scale_first,
            concat_params=concat_params,
            inverse=True,
            inner=inner,
        )



    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, setting):
        return projected_gradient_(H=H, y=y)


class PowellRestart(Module):
    """Powell's two restarting criterions for conjugate gradient methods.

    This should be placed **after** a conjugate gradient module and **before** the line search, other types of modules can also be used.

    The restart clears all states of ``modules``.

    Args:
        cond1 (float | None, optional):
            criterion that checks for nonconjugacy of the search directions.
            Restart is performed whenevr g^Tg_{k+1} >= cond1*||g_{k+1}||^2.
            The default condition value of 0.2 is suggested by Powell. Can be None to disable that criterion.
        cond2 (float | None, optional):
            criterion that checks if direction is not effectively downhill.
            Restart is performed if -1.2||g||^2 < d^Tg < -0.8||g||^2.
            Defaults to 0.2. Can be None to disable that criterion.

    Reference:
        Powell, Michael James David. "Restart procedures for the conjugate gradient method." Mathematical programming 12.1 (1977): 241-254.
    """
    def __init__(self, modules: Chainable, cond1:float | None = 0.2, cond2:float | None = 0.2):
        defaults=dict(cond1=cond1, cond2=cond2)
        super().__init__(defaults)

        self.set_child("modules", modules)

    def step(self, var):
        modules = self.children['modules']
        var = modules.step(var.clone(clone_update=False))

        g = TensorList(var.get_grad())
        if 'step' not in self.global_state:
            self.global_state['step'] = 0
            g_prev = self.get_state(var.params, 'prev_g', init=g)
            return var

        setting = self.settings[var.params[0]]
        cond1 = setting['cond1']; cond2 = setting['cond2']

        g_g = g.dot(g)

        if cond1 is not None:
            g_prev = self.get_state(var.params, 'prev_g', must_exist=True, cls=TensorList)
            g_g_prev = g_prev.dot(g)

            if g_g_prev.abs() >= cond1 * g_g:
                modules.reset()
                self.reset()
                var.update = g.clone()
                return var

        if cond2 is not None:
            d_g = TensorList(var.get_update()).dot(g)
            if (-1-cond2) * g_g < d_g < (-1 + cond2) * g_g:
                modules.reset()
                self.reset()
                var.update = g.clone()
                return var

        self.global_state['step'] = self.global_state['step'] + 1
        return var

class BirginMartinezRestart(Module):
    """the restart criterion for conjugate gradient methods designed by Birgin and Martinez.

    This criterion restarts when when the angle between dk+1 and −gk+1 is not acute enough.

    This should be placed **after** a conjugate gradient module and before the line search, other types of modules can also be used.

    The restart clears all states of ``modules``.

    Args:
        cond (float, optional):
            Restart is performed whenevr d^Tg > -cond*||d||*||g||.
            The default condition value of 1e-3 is suggested by Birgin and Martinez.

    Reference:
        Birgin, Ernesto G., and José Mario Martínez. "A spectral conjugate gradient method for unconstrained optimization." Applied Mathematics & Optimization 43.2 (2001): 117-128.
    """
    def __init__(self, modules: Chainable, cond:float = 1e-3):
        defaults=dict(cond=cond)
        super().__init__(defaults)

        self.set_child("modules", modules)

    def step(self, var):
        modules = self.children['modules']
        var = modules.step(var.clone(clone_update=False))

        setting = self.settings[var.params[0]]
        cond = setting['cond']

        g = TensorList(var.get_grad())
        d = TensorList(var.get_update())
        d_g = d.dot(g)
        d_norm = d.global_vector_norm()
        g_norm = g.global_vector_norm()

        if d_g > -cond * d_norm * g_norm:
            modules.reset()
            self.reset()
            var.update = g.clone()
            return var

        self.global_state['step'] = self.global_state['step'] + 1
        return var
