import math
from bisect import insort
from collections import deque
from collections.abc import Callable
from operator import itemgetter

import numpy as np
import torch

from .line_search import LineSearchBase, TerminationCondition, termination_condition


def adaptive_tracking(
    f,
    a_init,
    maxiter: int,
    nplus: float = 2,
    nminus: float = 0.5,
):
    niter = 0
    f_0 = f(0)

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


class AdaptiveTracking(LineSearchBase):
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
            'init', 'nplus', 'nminus', 'maxiter', 'adaptive')(self.settings[var.params[0]])

        objective = self.make_objective(var=var)

        # scale a_prev
        a_prev = self.global_state.get('a_prev', init)
        if adaptive: a_prev = a_prev * self.global_state.get('init_scale', 1)

        a_init = max(a_prev, 1e-10)
        step_size, f, niter = adaptive_tracking(
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

            self.global_state['a_prev'] = max(min(step_size, 1e16), 1e-10)
            return step_size

        # on fail reduce beta scale value
        self.global_state['init_scale'] = self.global_state.get('init_scale', 1) * nminus**maxiter
        self.global_state['a_prev'] = init
        return 0


def _within_bounds(x, lb, ub):
    if not math.isfinite(x): return False
    if lb is not None and x < lb: return False
    if ub is not None and x > ub: return False
    return True

def _quad_interp(f_0, g_0, a, f_a, bisect: bool, lb=None, ub=None) -> float | None:
    denom = 2 * (f_a - f_0 - g_0*a)
    if denom > 1e-10:
        num = g_0 * a**2
        a_min = num / -denom
        if _within_bounds(a_min, lb, ub): return a_min
    if bisect:
        return a/2
    return None


def _cubic_interp(f_0, g_0, a_1, f_1, a_2, f_2, bisect: bool, quad: bool, lb=None, ub=None) -> float | None:
    if a_1 > a_2:
        f_1, a_1, f_2, a_2 = f_2, a_2, f_1, a_1

    # ----------------------------------- cubic ---------------------------------- #
    v1 = (f_1 - f_0 - g_0 * a_1) / (a_1**2)
    v2 = (f_2 - f_0 - g_0 * a_2) / (a_2**2)

    denom = a_2 - a_1
    if abs(denom) > 1e-10:

        c3 = (v2 - v1) / denom
        c2 = v1 - a_1 * c3
        c1 = g_0

        if abs(c3) > 1e-10:
            discriminant = c2**2 - 3 * c3 * c1
            if discriminant > 0:
                a_min = (-c2 + np.sqrt(discriminant)) / (3 * c3)
                if 0 <= a_min <= a_2 and _within_bounds(a_min, lb, ub):
                    return a_min

        # --------------------------------- quadratic -------------------------------- #
        if not quad: return None
        w1 = (f_1 - f_0) / a_1
        w2 = (f_2 - f_0) / a_2

        b2 = (w2 - w1) / denom
        b1 = w1 - a_1 * b2

        if b2 > 1e-10:
            a_min = -b1 / (2 * b2)

            if 0 <= a_min <= a_2 and _within_bounds(a_min, lb, ub):
                return a_min

    # --------------------------------- bisectin --------------------------------- #
    if not bisect: return None
    fs = [(0, f_0), (a_1, f_1), (a_2, f_2)]
    fs.sort(key = lambda x: x[1])
    a1 = fs[0][0]; a2 = fs[1][0]
    return a1 + (a2-a1)/2

def _insort_(a_list, f_list, a_new, f_new):
    improved = math.isfinite(f_new) and (len(a_list) < 2 or f_new < f_list[1])

    if math.isfinite(f_new):
        fs = list(zip(a_list, f_list))
        insort(fs, (a_new, f_new), key = lambda x: x[1])

    return improved

# def adaptive_interpolation(
#     f,
#     a_0,
#     g_0,
#     maxiter: int,
#     c: float = 1e-4,
#     condition: TerminationCondition = 'armijo',
# ):
#     f_0 = f(0)
#     def satisfies(f_a, a):
#         return termination_condition(condition, f_0=f_0, g_0=g_0, f_a=f_a, g_a=None, a=a, c=c)

#     a_list = []
#     f_list = []
#     a = a_0
#     f_a =  f(a)
#     _insort_(a_list, f_list, a, f_a)

#     bisect = False
#     def interpolate_(backtrack, lb, ub):
#         nonlocal bisect
#         if bisect or len(a_list) == 0:
#             nonlocal a
#             if backtrack: a = a / 2
#             else: a = a * 2
#             if not _within_bounds(a, lb, ub): return False
#             f_a =  f(a)
#             _insort_(a_list, f_list, a, f_a)
#             if (not math.isfinite(f_a)) and (not backtrack): return False
#             return True

#         if len(a_list) == 1:
#             a_1 = a_list[0]; f_1 = f_list[0]
#             a_quad = _quad_interp(f_0, g_0, a_1, f_1, bisect=True, lb=lb, ub=ub)
#             if a_quad is None: return False
#             if not _within_bounds(a_quad, lb, ub): return False
#             f_a_quad =  f(a_quad)
#             _insort_(a_list, f_list, a_quad, f_a_quad)
#             if not math.isfinite(f_a_quad): bisect = True
#             return True

#         a_1, a_2 = a_list[:2]
#         f_1, f_2 = f_list[:2]
#         a_cub = _cubic_interp(f_0, g_0, a_1, f_1, a_2, f_2, bisect=True, quad=True, lb=lb, ub=ub)
#         if a_cub is None: return False
#         if not _within_bounds(a_cub, lb, ub): return False
#         f_a_cub =  f(a_cub)
#         _insort_(a_list, f_list, a_cub, f_a_cub)
#         if not math.isfinite(f_a_cub): bisect = True
#         return True


#     # backtrack
#     if f_a > f_0 or not math.isfinite(f_a):
#         while not satisfies(f_a, a):
#             maxiter -= 1
#             if maxiter < 0: return 0, f_0
#             success = interpolate_(backtrack=True, lb=0, ub=f_a)
#             if not success: return 0, f_0
#             if len(a_list) > 0:
#                 a = a_list[0]
#                 f_a = f_list[0]
#         return a, f_a

#     # forwardtrack
#     f_prev = f_a_best = f_a
#     a_prev = a_best = a
#     sat_prev = satisfies(f_a, a)
#     success = interpolate_(backtrack=False, lb=0, ub=None)
#     if not success:
#         if sat_prev: return a, f_a
#         return 0, f_0

#     if len(a_list) > 0:
#         a = a_list[0]
#         f_a = f_list[0]

#     # if f_prev < f_a:
#     #     if satisfies(f_prev, a_prev): return a_prev, f_prev
#     #     return 0, f_0

#         sat_prev = satisfies(f_a, a)
#         if sat_prev:
#             a_best = a
#             f_a_best = f_a

#     while f_prev >= f_a and math.isfinite(f_a):
#         maxiter -= 1
#         if maxiter < 0: break
#         f_prev = f_a
#         a_prev = a
#         success = interpolate_(backtrack=False, lb=0, ub=None)
#         if not success: break
#         if len(a_list) > 0:
#             a = a_list[0]
#             f_a = f_list[0]
#             sat = satisfies(f_a, a)
#             if sat:
#                 if f_a_best is None or f_a <= f_a_best:
#                     a_best = a
#                     f_a_best = f_a
#             if sat_prev and not sat:
#                 return a_prev, f_prev
#             sat_prev = sat

#     if satisfies(f_a_best, a_best): return a_best, f_a_best
#     return 0, f_0


# class AdaptiveInterpolation(LineSearchBase):
#     """
#     """
#     def __init__(
#         self,
#         init: float = 1.0,
#         maxiter: int = 10,
#         c: float = 1e-4,
#         condition: TerminationCondition = 'armijo',
#         adaptive=True,
#     ):
#         defaults=dict(init=init,maxiter=maxiter,adaptive=adaptive,c=c,condition=condition)
#         super().__init__(defaults=defaults)
#         self.global_state['init_scale'] = 1.0

#     def reset(self):
#         super().reset()
#         self.global_state['init_scale'] = 1.0

#     @torch.no_grad
#     def search(self, update, var):
#         init, maxiter, adaptive, c, condition = itemgetter('init', 'maxiter', 'adaptive', 'c', 'condition')(self.settings[var.params[0]])

#         objective = self.make_objective(var=var)

#         # directional derivative
#         if c == 0: d = 0
#         else: d = -sum(t.sum() for t in torch._foreach_mul(var.get_grad(), var.get_update()))

#         # scale beta (beta is multiplicative and i think may be better than scaling initial step size)
#         init_scale = self.global_state.get('init_scale', 1)
#         a_prev = self.global_state.get('a_prev', init)

#         if adaptive: a_prev = a_prev * init_scale

#         step_size, f = adaptive_interpolation(
#             objective,
#             a_0=a_prev,
#             g_0=d,
#             maxiter=maxiter,
#             c=c,
#             condition=condition,
#         )

#         # found an alpha that reduces loss
#         if step_size != 0:
#             self.global_state['init_scale'] = min(1.0, self.global_state.get('init_scale', 1) * math.sqrt(1.5))
#             self.global_state['a_prev'] = max(min(step_size, 1e16), 1e-10)
#             return step_size

#         # on fail reduce beta scale value
#         self.global_state['init_scale'] /= 1.5
#         self.global_state['a_prev'] = init
#         return 0
