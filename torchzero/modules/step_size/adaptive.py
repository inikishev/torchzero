"""Various step size strategies"""
from typing import Any, Literal
from operator import itemgetter
import torch

from ...core import Transform, Chainable
from ...utils import TensorList, unpack_dicts, unpack_states, NumberList, tofloat
from ...utils.linalg.linear_operator import ScaledIdentity
from ..functional import safe_scaling_

class PolyakStepSize(Transform):
    """Polyak's subgradient method with known or unknown f*.

    Args:
        f_star (float | Mone, optional):
            minimal possible value of the objective function. If not known, set to ``None``. Defaults to 0.
        y (float, optional):
            when ``f_star`` is set to None, it is calculated as ``f_best - y``.
        y_decay (float, optional):
            ``y`` is multiplied by ``(1 - y_decay)`` after each step. Defaults to 1e-3.
        max (float | None, optional): maximum possible step size. Defaults to None.
        use_grad (bool, optional):
            if True, uses dot product of update and gradient to compute the step size.
            Otherwise, dot product of update with itself is used.
        alpha (float, optional): multiplier to Polyak step-size. Defaults to 1.
    """
    def __init__(self, f_star: float | None = 0, y: float = 1, y_decay: float = 1e-3, max: float | None = None, use_grad=True, alpha: float = 1, inner: Chainable | None = None):

        defaults = dict(alpha=alpha, max=max, f_star=f_star, y=y, y_decay=y_decay)
        super().__init__(defaults, uses_grad=use_grad, uses_loss=True, inner=inner)

    @torch.no_grad
    def update_tensors(self, tensors, params, grads, loss, states, settings):
        assert grads is not None and loss is not None
        tensors = TensorList(tensors)
        grads = TensorList(grads)

        # load variables
        max, f_star, y, y_decay = itemgetter('max', 'f_star', 'y', 'y_decay')(settings[0])
        y_val = self.global_state.get('y_val', y)
        f_best = self.global_state.get('f_best', None)

        # gg
        if self._uses_grad: gg = tensors.dot(grads)
        else: gg = tensors.dot(tensors)

        # store loss
        if f_best is None or loss < f_best: f_best = tofloat(loss)
        if f_star is None: f_star = f_best - y_val

        # calculate the step size
        if gg <= torch.finfo(gg.dtype).eps: step_size = 0 # converged
        else: step_size = (loss - f_star) / gg

        # clip
        if max is not None:
            if step_size > max: step_size = max

        # store state
        self.global_state['f_best'] = f_best
        self.global_state['y_val'] = y_val * (1 - y_decay)
        self.global_state['step_size'] = step_size

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step_size = self.global_state.get('step_size', 1)
        torch._foreach_mul_(tensors, step_size * unpack_dicts(settings, 'alpha', cls=NumberList))
        return tensors

    def get_H(self, var):
        n = sum(p.numel() for p in var.params)
        p = var.params[0]
        return ScaledIdentity(self.global_state.get('step_size', 1), shape=(n,n), device=p.device, dtype=p.dtype)


def _bb_short(s: TensorList, y: TensorList, sy, eps):
    yy = y.dot(y)
    if yy < eps:
        if sy < eps: return None # try to fallback on long
        ss = s.dot(s)
        return ss/sy
    return sy/yy

def _bb_long(s: TensorList, y: TensorList, sy, eps):
    ss = s.dot(s)
    if sy < eps:
        yy = y.dot(y) # try to fallback on short
        if yy < eps: return None
        return sy/yy
    return ss/sy

def _bb_geom(s: TensorList, y: TensorList, sy, eps, fallback:bool):
    short = _bb_short(s, y, sy, eps)
    long = _bb_long(s, y, sy, eps)
    if long is None or short is None:
        if fallback:
            if short is not None: return short
            if long is not None: return long
        return None
    return (short * long) ** 0.5

class BarzilaiBorwein(Transform):
    """Barzilai-Borwein step size method.

    Args:
        type (str, optional):
            one of "short" with formula sᵀy/yᵀy, "long" with formula sᵀs/sᵀy, or "geom" to use geometric mean of short and long.
            Defaults to "geom".
        scale_first (bool, optional):
            whether to make first step very small when previous gradient is not available. Defaults to True.
        fallback (float, optional): step size when denominator is less than 0 (will happen on negative curvature). Defaults to 1e-3.
        inner (Chainable | None, optional):
            step size will be applied to outputs of this module. Defaults to None.

    """
    def __init__(self, type: Literal['long', 'short', 'geom', 'geom-fallback'] = 'geom', use_grad=True, scale_first:bool=True, fallback:float=1e-4, alpha:float=1, inner:Chainable|None = None):
        defaults = dict(type=type, fallback=fallback, alpha=alpha)
        super().__init__(defaults, uses_grad=use_grad, scale_first=scale_first, inner=inner)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_g')
        self.global_state.pop('step', None)

    @torch.no_grad
    def update_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        prev_p, prev_g = unpack_states(states, tensors, 'prev_p', 'prev_g', cls=TensorList)
        setting = settings[0]
        type = setting['type']

        g = grads if self._uses_grad else tensors
        assert g is not None

        if step != 0:
            s = params-prev_p
            y = g-prev_g
            sy = s.dot(y)
            eps = torch.finfo(sy.dtype).eps

            if type == 'short': step_size = _bb_short(s, y, sy, eps)
            elif type == 'long': step_size = _bb_long(s, y, sy, eps)
            elif type == 'geom': step_size = _bb_geom(s, y, sy, eps, fallback=False)
            elif type == 'geom-fallback': step_size = _bb_geom(s, y, sy, eps, fallback=True)
            else: raise ValueError(type)

            if step_size is not None: self.global_state['step_size'] = step_size

        prev_p.copy_(params)
        prev_g.copy_(g)

    def get_H(self, var):
        n = sum(p.numel() for p in var.params)
        p = var.params[0]
        return ScaledIdentity(self.global_state.get('step_size', 1), shape=(n,n), device=p.device, dtype=p.dtype)

    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step_size = self.global_state.get('step_size', 1)
        alpha = unpack_dicts(settings, 'alpha', cls=NumberList)
        if step_size is None: step_size = unpack_dicts(settings, 'fallback', cls=NumberList)
        torch._foreach_mul_(tensors, alpha*step_size)
        return tensors


class BBStab(Transform):
    """Stabilized Barzilai-Borwein method (https://arxiv.org/abs/1907.06409).

    This clips the norm of the Barzilai-Borwein update by ``delta`, where ``delta`` can be adaptive if ``c`` is specified.

    Args:
        c (float, optional):
            adaptive delta parameter. If ``delta`` is set to None, first ``inf_iters`` updates are performed
            with non-stabilized Barzilai-Borwein step size. Then delta is set to norm of
            the update that had the smallest norm, and multiplied by ``c``. Defaults to 0.2.
        delta (float | None, optional):
            Barzilai-Borwein update is clipped to this value. Set to ``None`` to use an adaptive choice. Defaults to None.
        type (str, optional):
            one of "short" with formula sᵀy/yᵀy, "long" with formula sᵀs/sᵀy, or "geom" to use geometric mean of short and long.
            Defaults to "geom". Note that "long" corresponds to BB1stab and "short" to BB2stab,
            however I found that "geom" works really well.
        scale_first (bool, optional):
            whether to make first step very small when previous gradient is not available. Defaults to True.
        fallback (float, optional): step size when denominator is less than 0 (will happen on negative curvature). Defaults to 1e-3.
        inner (Chainable | None, optional):
            step size will be applied to outputs of this module. Defaults to None.

    """
    def __init__(
        self,
        c=0.2,
        delta:float | None = None,
        type: Literal["long", "short", "geom", "geom-fallback"] = "geom",
        use_grad=True,
        scale_first: bool = True,
        fallback: float = 1e-4,
        alpha: float = 1,
        inf_iters: int = 3,
        inner: Chainable | None = None,
    ):
        defaults = dict(type=type, fallback=fallback, alpha=alpha, c=c, delta=delta, inf_iters=inf_iters)
        super().__init__(defaults, uses_grad=use_grad, scale_first=scale_first, inner=inner)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_g')
        self.global_state.pop('step', None)

    @torch.no_grad
    def update_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        prev_p, prev_g = unpack_states(states, tensors, 'prev_p', 'prev_g', cls=TensorList)
        setting = settings[0]
        type = setting['type']
        c = setting['c']
        delta = setting['delta']
        inf_iters = setting['inf_iters']

        g = grads if self._uses_grad else tensors
        assert g is not None
        g = TensorList(g)

        if step != 0:
            s = params-prev_p
            y = g-prev_g
            sy = s.dot(y)
            eps = torch.finfo(sy.dtype).eps

            if type == 'short': step_size = _bb_short(s, y, sy, eps)
            elif type == 'long': step_size = _bb_long(s, y, sy, eps)
            elif type == 'geom': step_size = _bb_geom(s, y, sy, eps, fallback=False)
            elif type == 'geom-fallback': step_size = _bb_geom(s, y, sy, eps, fallback=True)
            else: raise ValueError(type)

            if step_size is not None:

                # adaptive delta
                if delta is None:
                    niters = self.global_state.get('niters', 0) # this accounts for skipped negative curvature steps
                    self.global_state['niters'] = niters + 1

                    if niters < inf_iters:
                        s_norm_min = self.global_state.get('s_norm_min', None)
                        if s_norm_min is None: s_norm_min = s.global_vector_norm()
                        else: s_norm_min = min(s_norm_min, s.global_vector_norm())
                        self.global_state['s_norm_min'] = s_norm_min
                        # first few steps use delta=inf, so delta remains None

                    else:
                        delta = c * self.global_state['s_norm_min']

                if delta is None: # delta is inf for first few steps
                    self.global_state['step_size'] = step_size

                # BBStab step size
                else:
                    a_stab = delta / g.global_vector_norm()
                    self.global_state['step_size'] = min(step_size, a_stab)

        prev_p.copy_(params)
        prev_g.copy_(g)

    def get_H(self, var):
        n = sum(p.numel() for p in var.params)
        p = var.params[0]
        return ScaledIdentity(self.global_state.get('step_size', 1), shape=(n,n), device=p.device, dtype=p.dtype)

    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step_size = self.global_state.get('step_size', 1)
        alpha = unpack_dicts(settings, 'alpha', cls=NumberList)
        if step_size is None: step_size = unpack_dicts(settings, 'fallback', cls=NumberList)
        torch._foreach_mul_(tensors, alpha*step_size)
        return tensors



class AdaptiveGD(Transform):
    def __init__(self, a_0:float | None = None):
        defaults = dict(a_0=a_0)
        super().__init__(defaults)

