import math
from collections.abc import Callable
from operator import itemgetter

import torch

from .line_search import LineSearchBase



def adaptive_tracking(
    f,
    x_0,
    maxiter: int,
    nplus: float = 2,
    nminus: float = 0.5,
):
    f_0 = f(0)

    t = x_0
    f_t = f(t)

    # backtrack
    if f_t > f_0:
        while f_t > f_0:
            maxiter -= 1
            if maxiter < 0: return 0, f_0
            t = t*nminus
            f_t = f(t)
        return t, f_t

    # forwardtrack
    f_prev = f_t
    t *= nplus
    f_t = f(t)
    if f_prev < f_t: return t / nplus, f_prev
    while f_prev >= f_t:
        maxiter -= 1
        if maxiter < 0: return t, f_t
        f_prev = f_t
        t *= nplus
        f_t = f(t)
    return t / nplus, f_prev

class AdaptiveLineSearch(LineSearchBase):
    """Adaptive line search, similar to backtracking but also has forward tracking mode.
    Currently doesn't check for weak curvature condition.

    Args:
        init (float, optional): initial step size. Defaults to 1.0.
        beta (float, optional): multiplies each consecutive step size by this value. Defaults to 0.5.
        maxiter (int, optional): Maximum line search function evaluations. Defaults to 10.
        adaptive (bool, optional):
            when enabled, if line search failed, beta size is reduced.
            Otherwise it is reset to initial value. Defaults to True.
    """
    def __init__(
        self,
        init: float = 1.0,
        nplus: float = 2,
        nminus: float = 0.5,
        maxiter: int = 10,
        adaptive=True,
    ):
        defaults=dict(init=init,nplus=nplus,nminus=nminus,maxiter=maxiter,adaptive=adaptive,)
        super().__init__(defaults=defaults)
        self.global_state['beta_scale'] = 1.0

    def reset(self):
        super().reset()
        self.global_state['beta_scale'] = 1.0

    @torch.no_grad
    def search(self, update, var):
        init, nplus, nminus, maxiter, adaptive = itemgetter(
            'init', 'nplus', 'nminus', 'maxiter', 'adaptive')(self.settings[var.params[0]])

        objective = self.make_objective(var=var)

        # # directional derivative
        # d = -sum(t.sum() for t in torch._foreach_mul(var.get_grad(), var.get_update()))

        # scale beta (beta is multiplicative and i think may be better than scaling initial step size)
        beta_scale = self.global_state.get('beta_scale', 1)
        x_prev = self.global_state.get('prev_x', 1)

        if adaptive: nminus = nminus * beta_scale


        step_size, f = adaptive_tracking(objective, x_prev, maxiter, nplus=nplus, nminus=nminus)

        # found an alpha that reduces loss
        if step_size != 0:
            self.global_state['beta_scale'] = min(1.0, self.global_state['beta_scale'] * math.sqrt(1.5))
            return step_size

        # on fail reduce beta scale value
        self.global_state['beta_scale'] /= 1.5
        return 0
