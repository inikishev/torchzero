import math
from collections.abc import Callable
from operator import itemgetter

import torch

from .line_search import LineSearchBase, TerminationCondition, termination_condition



def adaptive_tracking(
    f,
    a_0,
    g_0,
    maxiter: int,
    nplus: float = 2,
    nminus: float = 0.5,
    c: float = 1e-4,
    condition: TerminationCondition = 'armijo',
):
    f_0 = f(0)

    def satisfies(f_a, a):
        return termination_condition(condition, f_0=f_0, g_0=g_0, f_a=f_a, g_a=None, a=a, c=c)

    a = a_0
    f_a = f(a)

    # backtrack
    if f_a > f_0 or not math.isfinite(f_a):
        while not satisfies(f_a, a):
            maxiter -= 1
            if maxiter < 0: return 0, f_0
            a = a*nminus
            f_a = f(a)
        return a, f_a

    # forwardtrack
    f_prev = f_a
    a_prev = a
    a *= nplus
    f_a = f(a)
    if f_prev < f_a or not math.isfinite(f_a):
        if satisfies(f_prev, a_prev): return a_prev, f_prev
        return 0, f_0

    sat_prev = satisfies(f_a, a)
    a_largest = None
    f_a_largest = None
    if sat_prev:
        a_largest = a
        f_a_largest = f_a

    while f_prev >= f_a and math.isfinite(f_a):
        maxiter -= 1
        if maxiter < 0: break
        f_prev = f_a
        a_prev = a
        a *= nplus
        f_a = f(a)
        sat = satisfies(f_a, a)
        if sat:
            a_largest = a
            f_a_largest = f_a
        if sat_prev and not sat:
            return a_prev, f_prev
        sat_prev = sat

    if a_largest is None: a_largest, f_a_largest = 0, f_0
    return a_largest, f_a_largest

class AdaptiveTracking(LineSearchBase):
    """Adaptive line search, similar to backtracking but also has forward tracking mode.

    Args:
        init (float, optional): initial step size. Defaults to 1.0.
        nplus (float, optional): multiplier to step size if initial step size is optimal. Defaults to 2.
        nminus (float, optional): multiplier to step size if initial step size is too big. Defaults to 0.5.
        maxiter (int, optional): maximum number of function evaluations per step. Defaults to 10.
        c (float, optional): termination condition value, type of condition depends on value of ``condition`` argument. Defaults to 1e-4.
        condition (TerminationCondition, optional):
            type of termination condition, only ones that do not use gradient at f(x+a*d) can be specified.
            - "armijo" - sufficient decrease condition.
            - "goldstein" - sufficient decrease plus second condition which prevents small step sizes, can be used with Newton-type methods.
            - "decrease" - any decrease in objective function value satisfies the condition.

            Defaults to 'armijo'.
        adaptive (bool, optional):
            when enabled, if line search failed, ``beta`` parameter is reduced by 1.5 times.
            Otherwise it is reset to initial value. Defaults to True.
    """
    def __init__(
        self,
        init: float = 1.0,
        nplus: float = 2,
        nminus: float = 0.5,
        maxiter: int = 10,
        c: float = 1e-4,
        condition: TerminationCondition = 'armijo',
        adaptive=True,
    ):
        defaults=dict(init=init,nplus=nplus,nminus=nminus,maxiter=maxiter,adaptive=adaptive,c1=c,condition=condition)
        super().__init__(defaults=defaults)
        self.global_state['beta_scale'] = 1.0

    def reset(self):
        super().reset()
        self.global_state['beta_scale'] = 1.0

    @torch.no_grad
    def search(self, update, var):
        init, nplus, nminus, maxiter, adaptive, c, condition = itemgetter(
            'init', 'nplus', 'nminus', 'maxiter', 'adaptive', 'c', 'condition')(self.settings[var.params[0]])

        objective = self.make_objective(var=var)

        # directional derivative
        if c == 0: d = 0
        else: d = -sum(t.sum() for t in torch._foreach_mul(var.get_grad(), var.get_update()))

        # scale beta (beta is multiplicative and i think may be better than scaling initial step size)
        beta_scale = self.global_state.get('beta_scale', 1)
        a_prev = self.global_state.get('a_prev', init)

        if adaptive: nminus = nminus * beta_scale

        step_size, f = adaptive_tracking(
            objective,
            a_0=a_prev,
            g_0=d,
            maxiter=maxiter,
            nplus=nplus,
            nminus=nminus,
            c=c,
            condition=condition,
        )

        # found an alpha that reduces loss
        if step_size != 0:
            self.global_state['beta_scale'] = min(1.0, self.global_state['beta_scale'] * math.sqrt(1.5))
            self.global_state['a_prev'] = max(min(step_size, 1e16), 1e-10)
            return step_size

        # on fail reduce beta scale value
        self.global_state['beta_scale'] /= 1.5
        self.global_state['a_prev'] = init
        return 0
