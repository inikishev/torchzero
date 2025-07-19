import math
from collections import deque

import numpy as np
import torch
from torch.optim.lbfgs import _cubic_interpolate

from ...core import Chainable, Transform
from ...utils import as_tensorlist
from ..line_search._polyinterp import polyinterp


def _isfinite(x):
    if isinstance(x, torch.Tensor): return torch.isfinite(x).all()
    return math.isfinite(x)

# based on https://github.com/pytorch/pytorch/blob/main/torch/optim/lbfgs.py
def _cubic_interpolate_unbounded(x1, f1, g1, x2, f2, g2):
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min_pos
    else:
        return None

class PolyStepSize(Transform):
    """Projects past points onto current update and fits a cubic or a polynomial, also could try putting ``tz.m.Normalize`` BEFORE this."""
    def __init__(self, order: int = 3, use_grad=True, scale_first:bool=True, init:float=1e-3, inner:Chainable|None = None):
        defaults = dict(order=order, scale_first=scale_first, init=init)
        super().__init__(defaults, uses_grad=use_grad, scale_first=scale_first, inner=inner, uses_loss=True)

    def _init_state(self, settings):
        order = settings[0]['order']
        self.global_state['f_history'] = deque(maxlen=order - 2)
        self.global_state['p_history'] = deque(maxlen=order - 2)
        self.global_state['g_history'] = deque(maxlen=order - 2)
        self.global_state['step'] = 0

    @torch.no_grad
    def update_tensors(self, tensors, params, grads, loss, states, settings):
        if 'step' not in self.global_state: self._init_state(settings)
        self.global_state['step'] += 1

        g = grads if self._uses_grad else tensors
        assert g is not None
        g = as_tensorlist(g)
        p = as_tensorlist(params)
        d = as_tensorlist(tensors)
        f = loss

        f_history = self.global_state['f_history']
        p_history = self.global_state['p_history']
        g_history = self.global_state['g_history']

        n = len(p_history)
        if n >= 1:
            t_min = None
            gd = g.dot(d)
            if gd > 1e-8:
                t1 = 0.
                f1 = f
                df1 = -gd

                # project previous points onto the current search direction
                ts = [(p - p_prev).dot(g) / gd for p_prev in p_history]
                fs = f_history
                dfs = [-g_prev.dot(d) for g_prev in g_history]

                if n == 1:
                    # use cubic interpolation
                    t2 = ts[0]
                    f2 = fs[0]
                    df2 = dfs[0]

                    if abs(t2) > 1e-9:
                        t_min = _cubic_interpolate_unbounded(t1, f1, df1, t2, f2, df2)

                else:
                    # polynomial interpolation
                    # polyinterp needs two-dimensional array with each point of form [x f g]
                    arr = torch.stack([torch.stack(ts), torch.stack(tuple(fs)), torch.stack(dfs)], -1).numpy(force=True)
                    try:
                        t_min = polyinterp(arr, x_min_bound = 0, x_max_bound=1e10)
                        if t_min > 1e9: t_min = None # means polyinterp fallbacked to bisection
                    except np.linalg.LinAlgError:
                        t_min = None

            if t_min is not None and _isfinite(t_min) and t_min > 0:
                step_size = float(t_min)
                self.global_state['step_size'] = step_size

        f_history.append(loss)
        p_history.append(p.clone())
        g_history.append(g.clone())

    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step_size = self.global_state.get('step_size', settings[0]['init'])
        torch._foreach_mul_(tensors, step_size)
        return tensors

