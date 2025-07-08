"""Various step size strategies"""
from typing import Any, Literal
from operator import itemgetter
import torch

from ...core import Transform, Chainable
from ...utils import TensorList, unpack_dicts, unpack_states, NumberList


class PolyakStepSize(Transform):
    """Polyak's subgradient method.

    Args:
        f_star (int, optional):
            (estimated) minimal possible value of the objective function (lowest possible loss). Defaults to 0.
        max (float | None, optional): maximum possible step size. Defaults to None.
        use_grad (bool, optional):
            if True, uses dot product of update and gradient to compute the step size.
            Otherwise, dot product of update with itself is used, which has no geometric meaning so it probably won't work well.
            Defaults to False.
        alpha (float, optional): multiplier to Polyak step-size. Defaults to 1.
    """
    def __init__(self, f_star: float = 0, max: float | None = None, use_grad=False, alpha: float = 1, inner: Chainable | None = None):

        defaults = dict(alpha=alpha, max=max, f_star=f_star, use_grad=use_grad)
        super().__init__(defaults, uses_grad=use_grad, uses_loss=True, inner=inner)

    def update_tensors(self, tensors, params, grads, loss, states, settings):
        assert grads is not None and loss is not None
        tensors = TensorList(tensors)
        grads = TensorList(grads)

        use_grad, max, f_star = itemgetter('use_grad', 'max', 'f_star')(settings[0])

        if use_grad: gg = tensors.dot(grads)
        else: gg = tensors.dot(tensors)

        if gg.abs() <= torch.finfo(gg.dtype).eps: step_size = 0 # converged
        else: step_size = (loss - f_star) / gg

        if max is not None:
            if step_size > max: step_size = max

        self.global_state['step_size'] = step_size

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step_size = self.global_state.get('step_size', 1)
        torch._foreach_mul_(tensors, step_size * unpack_dicts(settings, 'alpha', cls=NumberList))
        return tensors



def _bb_short(s: TensorList, y: TensorList, sy, eps, fallback):
    yy = y.dot(y)
    if yy < eps:
        if sy < eps: return fallback # try to fallback on long
        ss = s.dot(s)
        return ss/sy
    return sy/yy

def _bb_long(s: TensorList, y: TensorList, sy, eps, fallback):
    ss = s.dot(s)
    if sy < eps:
        yy = y.dot(y) # try to fallback on short
        if yy < eps: return fallback
        return sy/yy
    return ss/sy

def _bb_geom(s: TensorList, y: TensorList, sy, eps, fallback):
    short = _bb_short(s, y, sy, eps, fallback)
    long = _bb_long(s, y, sy, eps, fallback)
    return (short * long) ** 0.5

class BarzilaiBorwein(Transform):
    """Barzilai-Borwein method.

    Args:
        type (str, optional):
            one of "short" with formula sᵀy/yᵀy, "long" with formula sᵀs/sᵀy, or "geom" to use geometric mean of short and long.
            Defaults to 'geom'.
        scale_first (bool, optional):
            whether to make first step very small when previous gradient is not available. Defaults to True.
        fallback (float, optional): step size when denominator is less than 0 (will happen on negative curvature). Defaults to 1e-3.
        inner (Chainable | None, optional):
            step size will be applied to outputs of this module. Defaults to None.

    """
    def __init__(self, type: Literal['long', 'short', 'geom'] = 'geom', scale_first:bool=True, fallback:float=1e-3, inner:Chainable|None = None):
        defaults = dict(type=type, fallback=fallback)
        super().__init__(defaults, uses_grad=False, scale_first=scale_first, inner=inner)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_p', 'prev_g')

    @torch.no_grad
    def update_tensors(self, tensors, params, grads, loss, states, settings):
        prev_p, prev_g = unpack_states(states, tensors, 'prev_p', 'prev_g', cls=TensorList)
        fallback = unpack_dicts(settings, 'fallback', cls=NumberList)
        type = settings[0]['type']

        s = params-prev_p
        y = tensors-prev_g
        sy = s.dot(y)
        eps = torch.finfo(sy.dtype).eps

        if type == 'short': step_size = _bb_short(s, y, sy, eps, fallback)
        elif type == 'long': step_size = _bb_long(s, y, sy, eps, fallback)
        elif type == 'geom': step_size = _bb_geom(s, y, sy, eps, fallback)
        else: raise ValueError(type)

        self.global_state['step_size'] = step_size

        prev_p.copy_(params)
        prev_g.copy_(tensors)

    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step_size = self.global_state.get('step_size', 1)
        torch._foreach_mul_(tensors, step_size)
        return tensors

