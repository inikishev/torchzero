from abc import ABC, abstractmethod
from typing import Literal

import torch

from ...core import Chainable, TensorwiseTransform, Transform, apply_transform
from ...utils import TensorList, as_tensorlist, unpack_dicts, unpack_states


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

    def initialize(self, p: TensorList, g: TensorList):
        """runs on first step when prev_grads and prev_dir are not available"""

    @abstractmethod
    def get_beta(self, p: TensorList, g: TensorList, prev_g: TensorList, prev_d: TensorList) -> float | torch.Tensor:
        """returns beta"""

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        tensors = as_tensorlist(tensors)
        params = as_tensorlist(params)

        step = self.global_state.get('step', 0) + 1
        self.global_state['step'] = step
        prev_dir, prev_grads = unpack_states(states, tensors, 'prev_dir', 'prev_grad', cls=TensorList)

        # initialize on first step
        if step == 1:
            self.initialize(params, tensors)
            prev_dir.copy_(tensors)
            prev_grads.copy_(tensors)
            return tensors

        # get beta
        beta = self.get_beta(params, tensors, prev_grads, prev_dir)
        if settings[0]['clip_beta']: beta = max(0, beta) # pyright:ignore[reportArgumentType]
        prev_grads.copy_(tensors)

        # inner step
        if 'inner' in self.children:
            tensors = as_tensorlist(apply_transform(self.children['inner'], tensors, params, grads))

        # calculate new direction with beta
        dir = tensors.add_(prev_dir.mul_(beta))
        prev_dir.copy_(dir)

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
class ConjugateDescent(Transform):
    """Conjugate Descent (CD).

    .. note::
        - This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe(c2=0.1)` after this.
    """
    def __init__(self, inner: Chainable | None = None):
        super().__init__(defaults={}, uses_grad=False)

        if inner is not None:
            self.set_child('inner', inner)


    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        g = as_tensorlist(tensors)

        prev_d = unpack_states(states, tensors, 'prev_dir', cls=TensorList, init=torch.zeros_like)
        if 'denom' not in self.global_state:
            self.global_state['denom'] = torch.tensor(0.).to(g[0])

        prev_gd = self.global_state.get('prev_gd', 0)
        if abs(prev_gd) <= torch.finfo(g[0].dtype).eps: beta = 0
        else: beta = g.dot(g) / prev_gd

        # inner step
        if 'inner' in self.children:
            g = as_tensorlist(apply_transform(self.children['inner'], g, params, grads))

        dir = g.add_(prev_d.mul_(beta))
        prev_d.copy_(dir)
        self.global_state['prev_gd'] = g.dot(dir)
        return dir


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


# ----------------------------------- HS-DY ---------------------------------- #
def hs_dy_beta(g: TensorList, prev_d: TensorList,prev_g: TensorList):
    grad_diff = g - prev_g
    denom = prev_d.dot(grad_diff)
    if denom.abs() <= torch.finfo(g[0].dtype).eps: return 0

    # Dai-Yuan
    dy_beta = (g.dot(g) / denom).neg().clamp(min=0)

    # Hestenes–Stiefel
    hs_beta = (g.dot(grad_diff) / denom).neg().clamp(min=0)

    return max(0, min(dy_beta, hs_beta)) # type:ignore

class HybridHS_DY(ConguateGradientBase):
    """HS-DY hybrid conjugate gradient method.

    .. note::
        - This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe(c2=0.1)` after this.
    """
    def __init__(self, reset_interval: int | None | Literal['auto'] = None, clip_beta=False, inner: Chainable | None = None):
        super().__init__(clip_beta=clip_beta, reset_interval=reset_interval, inner=inner)

    def get_beta(self, p, g, prev_g, prev_d):
        return hs_dy_beta(g, prev_d, prev_g)


def projected_gradient_(H:torch.Tensor, y:torch.Tensor, tol: float):
    Hy = H @ y
    denom = y.dot(Hy)
    if denom.abs() < tol: return H
    H -= (H @ y.outer(y) @ H) / denom
    return H

class ProjectedGradientMethod(TensorwiseTransform):
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
        tol: float = 1e-10,
        reset_interval: int | None | Literal['auto'] = 'auto',
        update_freq: int = 1,
        scale_first: bool = False,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(reset_interval=reset_interval, tol=tol)
        super().__init__(defaults, uses_grad=False, scale_first=scale_first, concat_params=concat_params, update_freq=update_freq, inner=inner)

    def update_tensor(self, tensor, param, grad, loss, state, settings):
        step = state.get('step', 0) + 1
        state['step'] = step
        reset_interval = settings['reset_interval']
        if reset_interval == 'auto': reset_interval = tensor.numel() + 1 # as recommended

        if ("H" not in state) or (reset_interval is not None and step % reset_interval == 0):
            state["H"] = torch.eye(tensor.numel(), device=tensor.device, dtype=tensor.dtype)
            state['g_prev'] = tensor.clone()
            return

        H = state['H']
        g_prev = state['g_prev']
        state['g_prev'] = tensor.clone()
        y = (tensor - g_prev).ravel()

        projected_gradient_(H, y, settings['tol'])

    def apply_tensor(self, tensor, param, grad, loss, state, settings):
        H = state['H']
        return (H @ tensor.view(-1)).view_as(tensor)

# ---------------------------- Shor’s r-algorithm ---------------------------- #
# def shor_r(B:torch.Tensor, y:torch.Tensor, gamma:float):
#     r = B.T @ y
#     r /= torch.linalg.vector_norm(r).clip(min=1e-8) # pylint:disable=not-callable

#     I = torch.eye(B.size(1), device=B.device, dtype=B.dtype)
#     return B @ (I - gamma*r.outer(r))

# this is supposed to be equivalent
def shor_r_(H:torch.Tensor, y:torch.Tensor, alpha:float):
    p = H@y
    #(1-y)^2 (ppT)/(pTq)
    term = p.outer(p).div_(p.dot(y).clip(min=1e-8))
    H.sub_(term, alpha=1-alpha**2)
    return H

class ShorR(TensorwiseTransform):
    """Shor’s r-algorithm.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is required.

    Reference:
        Burke, James V., Adrian S. Lewis, and Michael L. Overton. "The Speed of Shor's R-algorithm." IMA Journal of numerical analysis 28.4 (2008): 711-720.

        Ansari, Zafar A. Limited Memory Space Dilation and Reduction Algorithms. Diss. Virginia Tech, 1998.
    """
    def __init__(self, alpha=0.5, scale_first:bool=True, concat_params: bool=True, update_freq: int= 1, reset_interval: int | None | Literal['auto'] = None, inner: Chainable | None = None,):
        defaults = dict(alpha=alpha, reset_interval=reset_interval)
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, scale_first=scale_first, update_freq=update_freq,inner=inner)

    def update_tensor(self, tensor, param, grad, loss, state, settings):
        reset_interval = settings['reset_interval']
        step = state.get('step', 0) + 1
        state['step'] = step

        if reset_interval == 'auto': reset_interval = tensor.numel() + 1
        g = tensor.ravel()

        if ("H" not in state) or (reset_interval is not None and step % reset_interval == 0):
            state["H"] = torch.eye(g.numel(), device=g.device, dtype=g.dtype)
            state["g_prev"] = g.clone()
            return

        H = state["H"]
        g_prev = state["g_prev"]
        y = g - g_prev
        g_prev.copy_(g)

        state["H"] = shor_r_(H, y, settings['alpha'])

    def apply_tensor(self, tensor, param, grad, loss, state, settings):
        H = state["H"]
        # return (B @ (B.T @ tensor.ravel())).view_as(tensor)
        return (H @ tensor.ravel()).view_as(tensor)