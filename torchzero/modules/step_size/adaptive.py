"""Various step size strategies"""
import math
from operator import itemgetter
from typing import Any, Literal

import torch

from ...core import Chainable, TensorTransform
from ...utils import NumberList, TensorList, tofloat, unpack_dicts, unpack_states
from ...linalg.linear_operator import ScaledIdentity
from ..opt_utils import epsilon_step_size

def _acceptable_alpha(alpha, param:torch.Tensor):
    finfo = torch.finfo(param.dtype)
    if (alpha is None) or (alpha < finfo.tiny*2) or (not math.isfinite(alpha)) or (alpha > finfo.max/2):
        return False
    return True

def _get_scaled_identity_H(self: TensorTransform, var):
    n = sum(p.numel() for p in var.params)
    p = var.params[0]
    alpha = self.global_state.get('alpha', 1)
    if not _acceptable_alpha(alpha, p): alpha = 1

    return ScaledIdentity(1 / alpha, shape=(n,n), device=p.device, dtype=p.dtype)


class PolyakStepSize(TensorTransform):
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
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
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
        if gg <= torch.finfo(gg.dtype).tiny * 2: alpha = 0 # converged
        else: alpha = (loss - f_star) / gg

        # clip
        if max is not None:
            if alpha > max: alpha = max

        # store state
        self.global_state['f_best'] = f_best
        self.global_state['y_val'] = y_val * (1 - y_decay)
        self.global_state['alpha'] = alpha

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        alpha = self.global_state.get('alpha', 1)
        if not _acceptable_alpha(alpha, tensors[0]): alpha = epsilon_step_size(TensorList(tensors))

        torch._foreach_mul_(tensors, alpha * unpack_dicts(settings, 'alpha', cls=NumberList))
        return tensors

    def get_H(self, objective):
        return _get_scaled_identity_H(self, objective)


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

class BarzilaiBorwein(TensorTransform):
    """Barzilai-Borwein step size method.

    Args:
        type (str, optional):
            one of "short" with formula sᵀy/yᵀy, "long" with formula sᵀs/sᵀy, or "geom" to use geometric mean of short and long.
            Defaults to "geom".
        fallback (float, optional): step size when denominator is less than 0 (will happen on negative curvature). Defaults to 1e-3.
        inner (Chainable | None, optional):
            step size will be applied to outputs of this module. Defaults to None.
    """

    def __init__(
        self,
        type: Literal["long", "short", "geom", "geom-fallback"] = "geom",
        alpha_0: float = 1e-7,
        use_grad=True,
        inner: Chainable | None = None,
    ):
        defaults = dict(type=type, alpha_0=alpha_0)
        super().__init__(defaults, uses_grad=use_grad, inner=inner)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_g')
        self.global_state['reset'] = True

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        prev_p, prev_g = unpack_states(states, tensors, 'prev_p', 'prev_g', cls=TensorList)
        type = self.defaults['type']

        g = grads if self._uses_grad else tensors
        assert g is not None

        reset = self.global_state.get('reset', False)
        self.global_state.pop('reset', None)

        if step != 0 and not reset:
            s = params-prev_p
            y = g-prev_g
            sy = s.dot(y)
            eps = torch.finfo(sy.dtype).tiny * 2

            if type == 'short': alpha = _bb_short(s, y, sy, eps)
            elif type == 'long': alpha = _bb_long(s, y, sy, eps)
            elif type == 'geom': alpha = _bb_geom(s, y, sy, eps, fallback=False)
            elif type == 'geom-fallback': alpha = _bb_geom(s, y, sy, eps, fallback=True)
            else: raise ValueError(type)

            # if alpha is not None:
            self.global_state['alpha'] = alpha

        prev_p.copy_(params)
        prev_g.copy_(g)

    def get_H(self, objective):
        return _get_scaled_identity_H(self, objective)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        alpha = self.global_state.get('alpha', None)

        if not _acceptable_alpha(alpha, tensors[0]):
            alpha = epsilon_step_size(TensorList(tensors), settings[0]['alpha_0'])

        torch._foreach_mul_(tensors, alpha)
        return tensors


class BBStab(TensorTransform):
    """Stabilized Barzilai-Borwein method (https://arxiv.org/abs/1907.06409).

    This clips the norm of the Barzilai-Borwein update by ``delta``, where ``delta`` can be adaptive if ``c`` is specified.

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
        inner (Chainable | None, optional):
            step size will be applied to outputs of this module. Defaults to None.

    """
    def __init__(
        self,
        c=0.2,
        delta:float | None = None,
        type: Literal["long", "short", "geom", "geom-fallback"] = "geom",
        alpha_0: float = 1e-7,
        use_grad=True,
        inf_iters: int = 3,
        inner: Chainable | None = None,
    ):
        defaults = dict(type=type,alpha_0=alpha_0, c=c, delta=delta, inf_iters=inf_iters)
        super().__init__(defaults, uses_grad=use_grad, inner=inner)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_g')
        self.global_state['reset'] = True

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        prev_p, prev_g = unpack_states(states, tensors, 'prev_p', 'prev_g', cls=TensorList)
        type = self.defaults['type']
        c = self.defaults['c']
        delta = self.defaults['delta']
        inf_iters = self.defaults['inf_iters']

        g = grads if self._uses_grad else tensors
        assert g is not None
        g = TensorList(g)

        reset = self.global_state.get('reset', False)
        self.global_state.pop('reset', None)

        if step != 0 and not reset:
            s = params-prev_p
            y = g-prev_g
            sy = s.dot(y)
            eps = torch.finfo(sy.dtype).tiny

            if type == 'short': alpha = _bb_short(s, y, sy, eps)
            elif type == 'long': alpha = _bb_long(s, y, sy, eps)
            elif type == 'geom': alpha = _bb_geom(s, y, sy, eps, fallback=False)
            elif type == 'geom-fallback': alpha = _bb_geom(s, y, sy, eps, fallback=True)
            else: raise ValueError(type)

            if alpha is not None:

                # adaptive delta
                if delta is None:
                    niters = self.global_state.get('niters', 0) # this accounts for skipped negative curvature steps
                    self.global_state['niters'] = niters + 1


                    if niters == 0: pass # 1st iteration is scaled GD step, shouldn't be used to find s_norm_min
                    elif niters <= inf_iters:
                        s_norm_min = self.global_state.get('s_norm_min', None)
                        if s_norm_min is None: s_norm_min = s.global_vector_norm()
                        else: s_norm_min = min(s_norm_min, s.global_vector_norm())
                        self.global_state['s_norm_min'] = s_norm_min
                        # first few steps use delta=inf, so delta remains None

                    else:
                        delta = c * self.global_state['s_norm_min']

                if delta is None: # delta is inf for first few steps
                    self.global_state['alpha'] = alpha

                # BBStab step size
                else:
                    a_stab = delta / g.global_vector_norm()
                    self.global_state['alpha'] = min(alpha, a_stab)

        prev_p.copy_(params)
        prev_g.copy_(g)

    def get_H(self, objective):
        return _get_scaled_identity_H(self, objective)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        alpha = self.global_state.get('alpha', None)

        if not _acceptable_alpha(alpha, tensors[0]):
            alpha = epsilon_step_size(TensorList(tensors), settings[0]['alpha_0'])

        torch._foreach_mul_(tensors, alpha)
        return tensors


class AdGD(TensorTransform):
    """AdGD and AdGD-2 (https://arxiv.org/abs/2308.02261)"""
    def __init__(self, variant:Literal[1,2]=2, alpha_0:float = 1e-7, sqrt:bool=True, use_grad=True, inner: Chainable | None = None,):
        defaults = dict(variant=variant, alpha_0=alpha_0, sqrt=sqrt)
        super().__init__(defaults, uses_grad=use_grad, inner=inner,)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_g')
        self.global_state['reset'] = True

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        variant = settings[0]['variant']
        theta_0 = 0 if variant == 1 else 1/3
        theta = self.global_state.get('theta', theta_0)

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        p = TensorList(params)
        g = grads if self._uses_grad else tensors
        assert g is not None
        g = TensorList(g)

        prev_p, prev_g = unpack_states(states, tensors, 'prev_p', 'prev_g', cls=TensorList)

        # online
        if self.global_state.get('reset', False):
            del self.global_state['reset']
            prev_p.copy_(p)
            prev_g.copy_(g)
            return

        if step == 0:
            alpha_0 = settings[0]['alpha_0']
            if alpha_0 is None: alpha_0 = epsilon_step_size(g)
            self.global_state['alpha']  = alpha_0
            prev_p.copy_(p)
            prev_g.copy_(g)
            return

        sqrt = settings[0]['sqrt']
        alpha = self.global_state.get('alpha', math.inf)
        L = (g - prev_g).global_vector_norm() / (p - prev_p).global_vector_norm()
        eps = torch.finfo(L.dtype).tiny * 2

        if variant == 1:
            a1 = math.sqrt(1 + theta)*alpha
            val = math.sqrt(2) if sqrt else 2
            if L > eps: a2 = 1 / (val*L)
            else: a2 = math.inf

        elif variant == 2:
            a1 = math.sqrt(2/3 + theta)*alpha
            a2 = alpha / math.sqrt(max(eps, 2 * alpha**2 * L**2 - 1))

        else:
            raise ValueError(variant)

        alpha_new = min(a1, a2)
        if alpha_new < 0: alpha_new = max(a1, a2)
        if alpha_new > eps:
            self.global_state['theta'] = alpha_new/alpha
            self.global_state['alpha'] = alpha_new

        prev_p.copy_(p)
        prev_g.copy_(g)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        alpha = self.global_state.get('alpha', None)

        if not _acceptable_alpha(alpha, tensors[0]):
            # alpha isn't None on 1st step
            self.state.clear()
            self.global_state.clear()
            alpha = epsilon_step_size(TensorList(tensors), settings[0]['alpha_0'])

        torch._foreach_mul_(tensors, alpha)
        return tensors

    def get_H(self, objective):
        return _get_scaled_identity_H(self, objective)
