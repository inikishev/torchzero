import math
import warnings
from operator import itemgetter
from typing import Literal

import numpy as np
import torch
from torch.optim.lbfgs import _cubic_interpolate

from ...utils import as_tensorlist, totensor, tofloat
from ._polyinterp import polyinterp, polyinterp2
from .line_search import LineSearchBase, TerminationCondition, termination_condition
from ..step_size.adaptive import _bb_geom

def _totensor(x):
    if not isinstance(x, torch.Tensor): return torch.tensor(x, dtype=torch.float32)
    return x

def _within_bounds(x, bounds):
    if bounds is None: return True
    lb,ub = bounds
    if lb is not None and x < lb: return False
    if ub is not None and x > ub: return False
    return True

def _apply_bounds(x, bounds):
    if bounds is None: return True
    lb,ub = bounds
    if lb is not None and x < lb: return lb
    if ub is not None and x > ub: return ub
    return x

class _StrongWolfe:
    def __init__(
        self,
        f,
        f_0,
        g_0,
        d_norm,
        a_init,
        a_max,
        c1,
        c2,
        maxiter,
        maxeval,
        maxzoom,
        tol_change,
        interpolation: Literal["quadratic", "cubic", "bisection", "polynomial", "polynomial2"],
    ):
        self._f = f
        self.f_0 = f_0
        self.g_0 = g_0
        self.d_norm = d_norm
        self.a_init = a_init
        self.a_max = a_max
        self.c1 = c1
        self.c2 = c2
        self.maxiter = maxiter
        if maxeval is None: maxeval = float('inf')
        self.maxeval = maxeval
        self.tol_change = tol_change
        self.num_evals = 0
        self.maxzoom = maxzoom
        self.interpolation = interpolation

        self.history = {}

    def f(self, a):
        if a in self.history: return self.history[a]
        self.num_evals += 1
        f_a, g_a = self._f(a)
        self.history[a] = (f_a, g_a)
        return f_a, g_a

    def interpolate(self, a_lo, f_lo, g_lo, a_hi, f_hi, g_hi, bounds=None):
        if self.interpolation == 'cubic':
            # pytorch cubic interpolate needs tensors
            a_lo = _totensor(a_lo); f_lo = _totensor(f_lo); g_lo = _totensor(g_lo)
            a_hi = _totensor(a_hi); f_hi = _totensor(f_hi); g_hi = _totensor(g_hi)
            return float(_cubic_interpolate(x1=a_lo, f1=f_lo, g1=g_lo, x2=a_hi, f2=f_hi, g2=g_hi, bounds=bounds))

        if self.interpolation == 'bisection':
            return _apply_bounds(a_lo + 0.5 * (a_hi - a_lo), bounds)

        if self.interpolation == 'quadratic':
            a = a_hi - a_lo
            denom = 2 * (f_hi - f_lo - g_lo*a)
            if denom > 1e-32:
                num = g_lo * a**2
                a_min = num / -denom
                return _apply_bounds(a_min, bounds)
            return _apply_bounds(a_lo + 0.5 * (a_hi - a_lo), bounds)

        if self.interpolation in ('polynomial', 'polynomial2'):
            finite_history = [(tofloat(a), tofloat(f), tofloat(g)) for a, (f,g) in self.history.items() if math.isfinite(a) and math.isfinite(f) and math.isfinite(g)]
            if bounds is None: bounds = (None, None)
            polyinterp_fn = polyinterp if self.interpolation == 'polynomial' else polyinterp2
            try:
                return  _apply_bounds(polyinterp_fn(np.array(finite_history), *bounds), bounds) # pyright:ignore[reportArgumentType]
            except torch.linalg.LinAlgError:
                return _apply_bounds(a_lo + 0.5 * (a_hi - a_lo), bounds)
        else:
            raise ValueError(self.interpolation)

    def zoom(self, a_lo, f_lo, g_lo, a_hi, f_hi, g_hi):
        if a_lo >= a_hi:
            a_hi, f_hi, g_hi, a_lo, f_lo, g_lo = a_lo, f_lo, g_lo, a_hi, f_hi, g_hi

        insuf_progress = False
        for _ in range(self.maxzoom):
            if self.num_evals >= self.maxeval: break
            if (a_hi - a_lo) * self.d_norm < self.tol_change: break # small bracket

            if not (math.isfinite(f_hi) and math.isfinite(g_hi)):
                a_hi = a_hi / 2
                f_hi, g_hi = self.f(a_hi)
                continue

            a_j = self.interpolate(a_lo, f_lo, g_lo, a_hi, f_hi, g_hi, bounds=(a_lo, min(a_hi, self.a_max)))

            # this part is from https://github.com/pytorch/pytorch/blob/main/torch/optim/lbfgs.py:
            eps = 0.1 * (a_hi - a_lo)
            if min(a_hi - a_j, a_j - a_lo) < eps:
                # interpolation close to boundary
                if insuf_progress or a_j >= a_hi or a_j <= a_lo:
                    # evaluate at 0.1 away from boundary
                    if abs(a_j - a_hi) < abs(a_j - a_lo):
                        a_j = a_hi - eps
                    else:
                        a_j = a_lo + eps
                    insuf_progress = False
                else:
                    insuf_progress = True
            else:
                insuf_progress = False

            f_j, g_j = self.f(a_j)

            if f_j > self.f_0 + self.c1*a_j*self.g_0 or f_j > f_lo:
                a_hi, f_hi, g_hi = a_j, f_j, g_j

            else:
                if abs(g_j) <= -self.c2 * self.g_0:
                    return a_j, f_j, g_j

                if g_j * (a_hi - a_lo) >= 0:
                    a_hi, f_hi, g_hi = a_lo, f_lo, g_lo

                a_lo, f_lo, g_lo = a_j, f_j, g_j

        # fail
        return None, None, None

    def search(self):
        a_i = min(self.a_init, self.a_max)
        f_i = g_i = None
        a_prev = 0
        f_prev = self.f_0
        g_prev = self.g_0
        for i in range(self.maxiter):
            if self.num_evals >= self.maxeval: break
            f_i, g_i = self.f(a_i)

            if f_i > self.f_0 + self.c1*a_i*self.g_0 or (i > 0 and f_i > f_prev):
                return self.zoom(a_prev, f_prev, g_prev, a_i, f_i, g_i)

            if abs(g_i) <= -self.c2 * self.g_0:
                return a_i, f_i, g_i

            if g_i >= 0:
                return self.zoom(a_i, f_i, g_i, a_prev, f_prev, g_prev)

            # from pytorch
            min_step = a_i + 0.01 * (a_i - a_prev)
            max_step = a_i * 10
            a_i_next = self.interpolate(a_prev, f_prev, g_prev, a_i, f_i, g_i, bounds=(min_step, min(max_step, self.a_max)))
            # a_i_next = self.interpolate(a_prev, f_prev, g_prev, a_i, f_i, g_i, bounds=(0, self.a_max))

            a_prev, f_prev, g_prev = a_i, f_i, g_i
            a_i = a_i_next

        if self.num_evals < self.maxeval:
            assert f_i is not None and g_i is not None
            return self.zoom(0, self.f_0, self.g_0, a_i, f_i, g_i)

        return None, None, None


class StrongWolfe(LineSearchBase):
    """Interpolation line search satisfying Strong Wolfe condition.

    Args:
        c1 (float, optional): sufficient descent condition. Defaults to 1e-4.
        c2 (float, optional): strong curvature condition. For CG set to 0.1. Defaults to 0.9.
        a_init (str, optional):
            strategy for initializing the initial step size guess.
            - "fixed" - uses a fixed value specified in `init_value` argument.
            - "first-order" - assumes first-order change in the function at iterate will be the same as that obtained at the previous step.
            - "quadratic" - interpolates quadratic to f(x_{-1}) and f_x.
            - "quadratic-clip" - same as quad, but uses min(1, 1.01*alpha) as described in Numerical Optimization.
            - "previous" - uses final step size found on previous iteration.

            For 2nd order methods it is usually best to leave at "fixed".
            For methods that do not produce well scaled search directions, e.g. conjugate gradient,
            "first-order" or "quadratic-clip" are recommended. Defaults to 'init'.
        a_max (float, optional): upper bound for the proposed step sizes. Defaults to 1e12.
        init_value (float, optional):
            initial step size. Used when ``a_init``="fixed", and with other strategies as fallback value. Defaults to 1.
        maxiter (int, optional): maximum number of line search iterations. Defaults to 25.
        maxzoom (int, optional): maximum number of zoom iterations. Defaults to 10.
        maxeval (int | None, optional): maximum number of function evaluations. Defaults to None.
        tol_change (float, optional): tolerance, terminates on small brackets. Defaults to 1e-9.
        interpolation (str, optional):
            What type of interpolation to use.
            - "bisection" - uses the middle point. This is robust, especially if the objective function is non-smooth, however it may need more function evaluations.
            - "quadratic" - minimizes a quadratic model, generally outperformed by "cubic".
            - "cubic" - minimizes a cubic model - this is the most widely used interpolation strategy.
            - "polynomial" - fits a a polynomial to all points obtained during line search.
            - "polynomial2" - alternative polynomial fit, where if a point is outside of bounds, a lower degree polynomial is tried.
            This may have faster convergence than "cubic" and "polynomial".

            Defaults to 'cubic'.
        adaptive (bool, optional):
            if True, the initial step size will be halved when line search failed to find a good direction.
            When a good direction is found, initial step size is reset to the original value. Defaults to True.
        fallback (bool, optional):
            if True, when no point satisfied strong wolfe criteria,
            returns a point with value lower than initial value that doesn't satisfy the criteria. Defaults to False.
        plus_minus (bool, optional):
            if True, enables the plus-minus variant, where if curvature is negative, line search is performed
            in the opposite direction. Defaults to False.


    ## Examples:

    Conjugate gradient method with strong wolfe line search. Nocedal, Wright recommend setting c2 to 0.1 for CG. Since CG doesn't produce well scaled directions, initial alpha can be determined from function values by ``a_init="first-order"``.

    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.PolakRibiere(),
        tz.m.StrongWolfe(c2=0.1, a_init="first-order")
    )
    ```

    LBFGS strong wolfe line search:
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.LBFGS(),
        tz.m.StrongWolfe()
    )
    ```

    """
    def __init__(
        self,
        c1: float = 1e-4,
        c2: float = 0.9,
        a_init: Literal['first-order', 'quadratic', 'quadratic-clip', 'previous', 'fixed'] = 'fixed',
        a_max: float = 1e12,
        init_value: float = 1,
        maxiter: int = 25,
        maxzoom: int = 10,
        maxeval: int | None = None,
        tol_change: float = 1e-9,
        interpolation: Literal["quadratic", "cubic", "bisection", "polynomial", 'polynomial2'] = 'cubic',
        adaptive = True,
        fallback:bool = False,
        plus_minus = False,
    ):
        defaults=dict(init_value=init_value,init=a_init,a_max=a_max,c1=c1,c2=c2,maxiter=maxiter,maxzoom=maxzoom, fallback=fallback,
                      maxeval=maxeval, adaptive=adaptive, interpolation=interpolation, plus_minus=plus_minus, tol_change=tol_change)
        super().__init__(defaults=defaults)

        self.global_state['initial_scale'] = 1.0

    @torch.no_grad
    def search(self, update, var):
        self._g_prev = self._f_prev = None
        objective = self.make_objective_with_derivative(var=var)

        init_value, init, c1, c2, a_max, maxiter, maxzoom, maxeval, interpolation, adaptive, plus_minus, fallback, tol_change = itemgetter(
            'init_value', 'init', 'c1', 'c2', 'a_max', 'maxiter', 'maxzoom',
            'maxeval', 'interpolation', 'adaptive', 'plus_minus', 'fallback', 'tol_change')(self.defaults)

        dir = as_tensorlist(var.get_updates())
        grad_list = var.get_grads()

        g_0 = -sum(t.sum() for t in torch._foreach_mul(grad_list, dir))
        f_0 = var.get_loss(False)
        dir_norm = dir.global_vector_norm()

        inverted = False
        if plus_minus and g_0 > 0:
            original_objective = objective
            def inverted_objective(a):
                l, g_a = original_objective(-a)
                return l, -g_a
            objective = inverted_objective
            inverted = True

        # --------------------- determine initial step size guess -------------------- #
        init = init.lower().strip()

        a_init = init_value
        if init == 'fixed':
            pass # use init_value

        elif init == 'previous':
            if 'a_prev' in self.global_state:
                a_init = self.global_state['a_prev']

        elif init == 'first-order':
            if 'g_prev' in self.global_state and g_0 < -torch.finfo(dir[0].dtype).tiny * 2:
                a_prev = self.global_state['a_prev']
                g_prev = self.global_state['g_prev']
                if g_prev < 0:
                    a_init = a_prev * g_prev / g_0

        elif init in ('quadratic', 'quadratic-clip'):
            if 'f_prev' in self.global_state and g_0 < -torch.finfo(dir[0].dtype).tiny * 2:
                f_prev = self.global_state['f_prev']
                if f_0 < f_prev:
                    a_init = 2 * (f_0 - f_prev) / g_0
                    if init == 'quadratic-clip': a_init = min(1, 1.01*a_init)
        else:
            raise ValueError(init)

        if adaptive:
            a_init *= self.global_state.get('initial_scale', 1)

        strong_wolfe = _StrongWolfe(
            f=objective,
            f_0=f_0,
            g_0=g_0,
            d_norm=dir_norm,
            a_init=a_init,
            a_max=a_max,
            c1=c1,
            c2=c2,
            maxiter=maxiter,
            maxzoom=maxzoom,
            maxeval=maxeval,
            tol_change=tol_change,
            interpolation=interpolation,
        )

        a, f_a, g_a = strong_wolfe.search()
        if inverted and a is not None: a = -a
        if f_a is not None and (f_a > f_0 or not math.isfinite(f_a)): a = None

        if fallback:
            if a is None or a==0 or not math.isfinite(a):
                lowest = min(strong_wolfe.history.items(), key=lambda x: x[1][0])
                if lowest[1][0] < f_0:
                    a = lowest[0]
                    f_a, g_a = lowest[1]
                    if inverted: a = -a

        if a is not None and a != 0 and math.isfinite(a):
            self.global_state['initial_scale'] = 1
            self.global_state['a_prev'] = a
            self.global_state['f_prev'] = f_0
            self.global_state['g_prev'] = g_0
            return a

        # fail
        if adaptive:
            self.global_state['initial_scale'] = self.global_state.get('initial_scale', 1) * 0.5
            finfo = torch.finfo(dir[0].dtype)
            if self.global_state['initial_scale'] < finfo.tiny * 2:
                self.global_state['initial_scale'] = init_value * 2

        return 0
