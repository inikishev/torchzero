from abc import ABC, abstractmethod

import torch

from ...core import Chainable, Transform, apply
from ...utils import TensorList, as_tensorlist


class ConguateGradientBase(Transform, ABC):
    """all CGs are the same except beta calculation"""
    def __init__(self, defaults = None, clip_beta: bool = False, reset_interval: int | None = None, inner: Chainable | None = None):
        if defaults is None: defaults = {}
        defaults['reset_interval'] = reset_interval
        defaults['clip_beta'] = clip_beta
        super().__init__(defaults, uses_grad=False)

        if inner is not None:
            self.set_child('inner', inner)

    def initialize(self, p: TensorList, g: TensorList):
        """runs on first step when prev_grads and prev_dir are not available"""

    @abstractmethod
    def get_beta(self, p: TensorList, g: TensorList, prev_g: TensorList, prev_d: TensorList) -> float | torch.Tensor:
        """returns beta"""

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        tensors = as_tensorlist(tensors)
        params = as_tensorlist(params)

        step = self.global_state.get('step', 0)
        prev_dir, prev_grads = self.get_state('prev_dir', 'prev_grad', params=params, cls=TensorList)

        # initialize on first step
        if step == 0:
            self.initialize(params, tensors)
            prev_dir.copy_(tensors)
            prev_grads.copy_(tensors)
            self.global_state['step'] = step + 1
            return tensors

        # get beta
        beta = self.get_beta(params, tensors, prev_grads, prev_dir)
        if self.settings[params[0]]['clip_beta']: beta = max(0, beta) # pyright:ignore[reportArgumentType]
        prev_grads.copy_(tensors)

        # inner step
        if 'inner' in self.children:
            tensors = as_tensorlist(apply(self.children['inner'], tensors, params, grads, vars))

        # calculate new direction with beta
        dir = tensors.add_(prev_dir.mul_(beta))
        prev_dir.copy_(dir)

        # resetting
        self.global_state['step'] = step + 1
        reset_interval = self.settings[params[0]]['reset_interval']
        if reset_interval is not None and (step+1) % reset_interval == 0:
            self.reset()

        return dir

# ------------------------------- Polak-Ribière ------------------------------ #
def polak_ribiere_beta(g: TensorList, prev_g: TensorList):
    denom = prev_g.dot(prev_g)
    if denom == 0: return 0
    return g.dot(g - prev_g) / denom

class PolakRibiere(ConguateGradientBase):
    """Polak-Ribière-Polyak nonlinear conjugate gradient method. This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe(c2=0.1)` after this."""
    def __init__(self, clip_beta=True, reset_interval: int | None = None, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return polak_ribiere_beta(g, prev_g)

# ------------------------------ Fletcher–Reeves ----------------------------- #
def fletcher_reeves_beta(gg, prev_gg):
    if prev_gg == 0: return 0
    return gg / prev_gg

class FletcherReeves(ConguateGradientBase):
    """Fletcher–Reeves nonlinear conjugate gradient method. This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self, reset_interval: int | None = None, clip_beta=False, inner: Chainable | None = None):
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
    if denom == 0: return 0
    return (g.dot(grad_diff) / denom).neg()


class HestenesStiefel(ConguateGradientBase):
    """Hestenes–Stiefel nonlinear conjugate gradient method. This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self, reset_interval: int | None = None, clip_beta=False, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return hestenes_stiefel_beta(g, prev_d, prev_g)


# --------------------------------- Dai–Yuan --------------------------------- #
def dai_yuan_beta(g: TensorList, prev_d: TensorList,prev_g: TensorList):
    denom = prev_d.dot(g - prev_g)
    if denom == 0: return 0
    return (g.dot(g) / denom).neg()

class DaiYuan(ConguateGradientBase):
    """Dai–Yuan nonlinear conjugate gradient method. This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self, reset_interval: int | None = None, clip_beta=False, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return dai_yuan_beta(g, prev_d, prev_g)


# -------------------------------- Liu-Storey -------------------------------- #
def liu_storey_beta(g:TensorList, prev_d:TensorList, prev_g:TensorList, ):
    denom = prev_g.dot(prev_d)
    if denom == 0: return 0
    return g.dot(g - prev_g) / denom

class LiuStorey(ConguateGradientBase):
    """Liu-Storey nonlinear conjugate gradient method. This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self, reset_interval: int | None = None, clip_beta=False, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return liu_storey_beta(g, prev_d, prev_g)

# ----------------------------- Conjugate Descent ---------------------------- #
class ConjugateDescent(Transform):
    """Conjugate Descent (CD). This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self, inner: Chainable | None = None):
        super().__init__(defaults={}, uses_grad=False)

        if inner is not None:
            self.set_child('inner', inner)


    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        g = as_tensorlist(tensors)

        prev_d = self.get_state('prev_dir', params=params, cls=TensorList, init = torch.zeros_like)
        if 'denom' not in self.global_state:
            self.global_state['denom'] = torch.tensor(0.).to(g[0])

        prev_gd = self.global_state.get('prev_gd', 0)
        if prev_gd == 0: beta = 0
        else: beta = g.dot(g) / prev_gd

        # inner step
        if 'inner' in self.children:
            g = as_tensorlist(apply(self.children['inner'], g, params, grads, vars))

        dir = g.add_(prev_d.mul_(beta))
        prev_d.copy_(dir)
        self.global_state['prev_gd'] = g.dot(dir)
        return dir


# -------------------------------- Hager-Zhang ------------------------------- #
def hager_zhang_beta(g:TensorList, prev_d:TensorList, prev_g:TensorList,):
    g_diff = g - prev_g
    denom = prev_d.dot(g_diff)
    if denom == 0: return 0

    term1 = 1/denom
    # term2
    term2 = (g_diff - (2 * prev_d * (g_diff.pow(2).global_sum()/denom))).dot(g)
    return (term1 * term2).neg()


class HagerZhang(ConguateGradientBase):
    """Hager-Zhang nonlinear conjugate gradient method,
    This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self, reset_interval: int | None = None, clip_beta=False, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return hager_zhang_beta(g, prev_d, prev_g)


# ----------------------------------- HS-DY ---------------------------------- #
def hs_dy_beta(g: TensorList, prev_d: TensorList,prev_g: TensorList):
    grad_diff = g - prev_g
    denom = prev_d.dot(grad_diff)
    if denom == 0: return 0

    # Dai-Yuan
    dy_beta = (g.dot(g) / denom).neg().clamp(min=0)

    # Hestenes–Stiefel
    hs_beta = (g.dot(grad_diff) / denom).neg().clamp(min=0)

    return max(0, min(dy_beta, hs_beta)) # type:ignore

class HybridHS_DY(ConguateGradientBase):
    """HS-DY hybrid conjugate gradient method.
    This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self, reset_interval: int | None = None, clip_beta=False, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return hs_dy_beta(g, prev_d, prev_g)
