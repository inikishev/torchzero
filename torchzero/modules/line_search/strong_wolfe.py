import math
import warnings
from operator import itemgetter

import torch
from torch.optim.lbfgs import _cubic_interpolate

from .line_search import LineSearch
from .backtracking import backtracking_line_search
from ...utils import totensor


def _zoom(objective,
          alpha_low, alpha_high,
          phi_low, phi_prime_low,
          phi_0, phi_prime_0, c1, c2,
          max_zoom_iter=10):

    phi_high, phi_prime_high = objective(alpha_high)

    for i in range(max_zoom_iter):
        alpha_j = _cubic_interpolate(
            *(totensor(i) for i in (alpha_low, phi_low, phi_prime_low, alpha_high, phi_high, phi_prime_high))
        )

        # if interpolation fails or produces endpoint, bisect
        delta = abs(alpha_high - alpha_low)
        if alpha_j is None or alpha_j == alpha_low or alpha_j == alpha_high:
            alpha_j = alpha_low + 0.5 * delta

        phi_j, phi_prime_j = objective(alpha_j)

        # check armijo
        armijo_satisfied = phi_j <= phi_0 + c1 * alpha_j * phi_prime_0

        # check strong wolfe
        strong_curvature_satisfied = abs(phi_prime_j) <= c2 * abs(phi_prime_0)


        # minimum between alpha_low and alpha_j
        if not armijo_satisfied or phi_j >= phi_low:
            alpha_high = alpha_j
            phi_high = phi_j
            phi_prime_high = phi_prime_j
        else:
            # alpha_j satisfies armijo
            if strong_curvature_satisfied:
                return alpha_j

            # minimum between alpha_j and alpha_high
            if phi_prime_j * (alpha_high - alpha_low) >= 0:
                # between alpha_low and alpha_j
                alpha_high = alpha_low
                phi_high = phi_low
                phi_prime_high = phi_prime_low

            alpha_low = alpha_j
            phi_low = phi_j
            phi_prime_low = phi_prime_j


        # check if interval too small
        if abs(alpha_high - alpha_low) < 1e-10 * max(alpha_low, alpha_high, 1.0):
             # check low and high ends if they satisfy condition
            if (phi_low <= phi_0 + c1 * alpha_low * phi_prime_0 and abs(phi_prime_low) <= c2 * abs(phi_prime_0)):
                return alpha_low
            # check high end
            elif (phi_high <= phi_0 + c1 * alpha_high * phi_prime_0 and abs(phi_prime_high) <= c2 * abs(phi_prime_0)):
                return alpha_high
            else:
                # alpha_low satisfies armijo so just use it
                return alpha_low

    if math.isnan(alpha_high): return None
    return None


def strong_wolfe(
    objective,
    initial_step_size: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_ls_iter: int = 25,
    alpha_max: float = 1e10,
    increase_factor: float = 2.0,  # Factor to increase alpha in bracketing
    plus_minus: bool = True,
) -> float | None:
    alpha_prev = 0.0
    phi_0, phi_prime_0 = objective(alpha_prev)

    if phi_prime_0 == 0: return 0
    if phi_prime_0 > 0:
        # if direction is not a descent direction, perform line search in opposite direction
        if plus_minus:
            def inverted_objective(alpha):
                l, g = objective(-alpha)
                return l, -g
            v = strong_wolfe(
                inverted_objective,
                initial_step_size=initial_step_size,
                c1=c1,
                c2=c2,
                max_ls_iter=max_ls_iter,
                alpha_max=alpha_max,
                increase_factor=increase_factor,
                plus_minus=plus_minus,
            )
            if v is not None: v = -v
            return v
        else: return 0

    phi_prev = phi_0
    phi_prime_prev = phi_prime_0
    alpha_curr = initial_step_size

    # bracket
    for i in range(max_ls_iter):

        phi_curr, phi_prime_curr = objective(alpha_curr)

        # potential fallback but when?
        # alpha_curr = (alpha_prev + alpha_curr) / 2.0 # Bisect
        # return _zoom(objective,
        #                 alpha_prev, alpha_curr, # alpha_curr is now the high bound
        #                 phi_prev, phi_prime_prev,
        #                 phi_0, phi_prime_0, c1, c2)


        # check armijo
        armijo_violated = phi_curr > phi_0 + c1 * alpha_curr * phi_prime_0
        func_increased = phi_curr >= phi_prev and i > 0

        if armijo_violated or func_increased:
            return _zoom(objective,
                         alpha_prev, alpha_curr,
                         phi_prev, phi_prime_prev,
                         phi_0, phi_prime_0, c1, c2)


        # check strong wolfe
        if abs(phi_prime_curr) <= c2 * abs(phi_prime_0):
            return alpha_curr

        # minimum is bracketed
        if phi_prime_curr >= 0:
            return _zoom(objective,
                        #alpha_curr, alpha_prev,
                        alpha_prev, alpha_curr,
                        #phi_curr, phi_prime_curr,
                        phi_prev, phi_prime_prev,
                        phi_0, phi_prime_0, c1, c2)

        # otherwise continue bracketing
        alpha_next = alpha_curr * increase_factor

        if alpha_next > alpha_max:
            alpha_curr = alpha_max
            phi_curr, phi_prime_curr = objective(alpha_curr)

            armijo_violated = phi_curr > phi_0 + c1 * alpha_curr * phi_prime_0
            strong_curvature_met = abs(phi_prime_curr) <= c2 * abs(phi_prime_0)

            if not armijo_violated and strong_curvature_met: # wolfe met at alpha_max
                return alpha_curr
            elif armijo_violated or (phi_curr >= phi_prev and i > 0): # armijo failed at alpha_max
                return _zoom(objective, alpha_prev, alpha_curr, phi_prev, phi_prime_prev, phi_0, phi_prime_0, c1, c2)
            elif phi_prime_curr >= 0:  # derivative positive at alpha_max
                return _zoom(
                    objective,
                    #alpha_curr, alpha_prev,
                    #phi_curr, phi_prime_curr,
                    alpha_prev, alpha_curr,
                    phi_prev, phi_prime_prev,
                    phi_0, phi_prime_0, c1, c2)
            else:
                return alpha_max

        else:
            # update previous point and continue loop with increased step size
            alpha_prev = alpha_curr
            phi_prev = phi_curr
            phi_prime_prev = phi_prime_curr
            alpha_curr = alpha_next


    # max iters reached
    return None

class StrongWolfe(LineSearch):
    def __init__(
        self,
        initial_step_size: float = 1.0,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_ls_iter: int = 25,
        alpha_max: float = 1e10,
        increase_factor: float = 2.0,
        adaptive = True,
        fallback = True,
        plus_minus = True,
    ):
        defaults=dict(initial_step_size=initial_step_size,c1=c1,c2=c2,max_ls_iter=max_ls_iter,
                      alpha_max=alpha_max,increase_factor=increase_factor, adaptive=adaptive, fallback=fallback, plus_minus=plus_minus)
        super().__init__(defaults=defaults)

        self.global_state['initial_scale'] = 1.0
        self.global_state['beta_scale'] = 1.0

    @torch.no_grad
    def search(self, update, vars):
        objective = self.make_objective_with_derivative(vars=vars)

        initial_step_size, c1, c2, max_ls_iter, alpha_max, increase_factor, adaptive, fallback, plus_minus = itemgetter(
            'initial_step_size', 'c1', 'c2', 'max_ls_iter', 'alpha_max',
            'increase_factor', 'adaptive', 'fallback', 'plus_minus')(self.settings[vars.params[0]])

        step_size = strong_wolfe(
            objective,
            initial_step_size=initial_step_size * self.global_state["initial_scale"],
            c1=c1,
            c2=c2,
            max_ls_iter=max_ls_iter,
            alpha_max=alpha_max,
            increase_factor=increase_factor,
            plus_minus=plus_minus,
        )

        if step_size is not None and step_size != 0:
            self.global_state['initial_scale'] = min(1.0, self.global_state['initial_scale'] * math.sqrt(2))
            return step_size

        # fallback to backtracking on fail
        if adaptive: self.global_state['initial_scale'] *= 0.5
        if not fallback: return 0

        objective = self.make_objective(vars=vars)

        # # directional derivative
        d = -sum(t.sum() for t in torch._foreach_mul(vars.get_grad(), vars.get_update()))

        step_size = backtracking_line_search(
            objective,
            d,
            initial_step_size=initial_step_size * self.global_state["initial_scale"],
            beta=0.5 * self.global_state["beta_scale"],
            c=1e-8,
            max_iter=max_ls_iter * 2,
            min_alpha=None,
        )

        # found an alpha that reduces loss
        if step_size is not None:
            self.global_state['beta_scale'] = min(1.0, self.global_state['beta_scale'] * math.sqrt(1.5))
            return step_size

        # on fail reduce beta scale value
        self.global_state['beta_scale'] /= 1.5
        return 0