import math
from bisect import insort
from collections import deque
from collections.abc import Callable
from operator import itemgetter

import numpy as np
import torch

from .line_search import LineSearchBase, TerminationCondition, termination_condition


def adaptive_bisection(
    f,
    a_init,
    maxiter: int,
    nplus: float = 2,
    nminus: float = 0.5,
    f_0 = None,
):
    niter = 0
    if f_0 is None: f_0 = f(0)

    a = a_init
    f_a = f(a)

    # backtrack
    a_prev = a
    f_prev = math.inf
    if (f_a > f_0) or (not math.isfinite(f_a)):
        while (f_a < f_prev) or not math.isfinite(f_a):
            a_prev, f_prev = a, f_a
            maxiter -= 1
            if maxiter < 0: break

            a = a*nminus
            f_a = f(a)
            niter += 1

        if f_prev < f_0: return a_prev, f_prev, niter
        return 0, f_0, niter

    # forwardtrack
    a_prev = a
    f_prev = math.inf
    while (f_a <= f_prev) and math.isfinite(f_a):
        a_prev, f_prev = a, f_a
        maxiter -= 1
        if maxiter < 0: break

        a *= nplus
        f_a = f(a)
        niter+= 1

    if f_prev < f_0: return a_prev, f_prev, niter
    return 0, f_0, niter


class AdaptiveBisection(LineSearchBase):
    """A line search that evaluates previous step size, if value increased, backtracks until the value stops decreasing,
    otherwise forward-tracks until value stops decreasing.

    Args:
        init (float, optional): initial step size. Defaults to 1.0.
        nplus (float, optional): multiplier to step size if initial step size is optimal. Defaults to 2.
        nminus (float, optional): multiplier to step size if initial step size is too big. Defaults to 0.5.
        maxiter (int, optional): maximum number of function evaluations per step. Defaults to 10.
        adaptive (bool, optional):
            when enabled, if line search failed, step size will continue decreasing on the next step.
            Otherwise it will restart the line search from ``init`` step size. Defaults to True.
    """
    def __init__(
        self,
        init: float = 1.0,
        nplus: float = 2,
        nminus: float = 0.5,
        maxiter: int = 10,
        adaptive=True,
    ):
        defaults=dict(init=init,nplus=nplus,nminus=nminus,maxiter=maxiter,adaptive=adaptive)
        super().__init__(defaults=defaults)

    def reset(self):
        super().reset()

    @torch.no_grad
    def search(self, update, var):
        init, nplus, nminus, maxiter, adaptive = itemgetter(
            'init', 'nplus', 'nminus', 'maxiter', 'adaptive')(self.defaults)

        objective = self.make_objective(var=var)

        # scale a_prev
        a_prev = self.global_state.get('a_prev', init)
        if adaptive: a_prev = a_prev * self.global_state.get('init_scale', 1)

        a_init = a_prev
        if a_init < torch.finfo(var.params[0].dtype).tiny * 2:
            a_init = torch.finfo(var.params[0].dtype).max / 2

        step_size, f, niter = adaptive_bisection(
            objective,
            a_init=a_init,
            maxiter=maxiter,
            nplus=nplus,
            nminus=nminus,
        )

        # found an alpha that reduces loss
        if step_size != 0:
            assert (var.loss is None) or (math.isfinite(f) and f < var.loss)
            self.global_state['init_scale'] = 1

            # if niter == 1, forward tracking failed to decrease function value compared to f_a_prev
            if niter == 1 and step_size >= a_init: step_size *= nminus

            self.global_state['a_prev'] = step_size
            return step_size

        # on fail reduce beta scale value
        self.global_state['init_scale'] = self.global_state.get('init_scale', 1) * nminus**maxiter
        self.global_state['a_prev'] = init
        return 0

