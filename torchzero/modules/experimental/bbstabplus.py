"""Various step size strategies"""
import math
from operator import itemgetter
from typing import Any, Literal

import torch

from ...core import Chainable, Transform
from ...utils import NumberList, TensorList, tofloat, unpack_dicts, unpack_states
from ...utils.linalg.linear_operator import ScaledIdentity
from ..functional import epsilon_step_size, initial_step_size
from ..step_size.adaptive import _bb_geom, _bb_long, _bb_short


class BBStabPlus(Transform):
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
        inner (Chainable | None, optional):
            step size will be applied to outputs of this module. Defaults to None.

    """
    def __init__(
        self,
        c=0.2,
        delta:float | None = None,
        type: Literal["long", "short", "geom", "geom-fallback"] = "geom",
        use_grad=True,
        inf_iters: int = 3,
        inner: Chainable | None = None,
    ):
        defaults = dict(type=type,c=c, delta=delta, inf_iters=inf_iters)
        super().__init__(defaults, uses_grad=use_grad, inner=inner)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_g')
        self.global_state['reset'] = True

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

        reset = self.global_state.get('reset', False)
        self.global_state.pop('reset', None)

        if step != 0 and not reset:
            s = params-prev_p
            y = g-prev_g
            sy = s.dot(y)
            eps = torch.finfo(sy.dtype).min

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

                    if niters < inf_iters:
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

    def get_H(self, var):
        n = sum(p.numel() for p in var.params)
        p = var.params[0]
        return ScaledIdentity(self.global_state.get('alpha', 1), shape=(n,n), device=p.device, dtype=p.dtype)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        alpha = self.global_state.get('alpha', None)

        if alpha is None or alpha < 0 or not math.isfinite(alpha):
            alpha = initial_step_size(TensorList(tensors))

        torch._foreach_mul_(tensors, alpha)
        return tensors