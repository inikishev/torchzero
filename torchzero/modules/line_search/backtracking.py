import math
from collections.abc import Callable
from operator import itemgetter

import torch

from .line_search import LineSearch


def backtracking_line_search(
    objective: Callable[[float], float],
    dir_derivative: float | torch.Tensor,
    initial_step_size: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
    max_iter: int = 10,
    min_alpha: float | None = None
) -> float | None:
    """

    Args:
        objective_fn: evaluates step size along some descent direction.
        dir_derivative: directional derivative along the descent direction.
        alpha_init: initial step size.
        beta: The factor by which to decrease alpha in each iteration
        c: The constant for the Armijo sufficient decrease condition
        max_iter: Maximum number of backtracking iterations (default: 10).
        min_alpha: Minimum allowable step size to prevent near-zero values (default: 1e-16).

    Returns:
        step size
    """

    alpha = initial_step_size
    f_x = objective(0)

    for iteration in range(max_iter):
        f_alpha = objective(alpha)
        sufficient_f = f_x + c * alpha * dir_derivative

        if f_alpha <= sufficient_f:
            # found an acceptable alpha
            return alpha
        else:
            # decrease alpha
            alpha *= beta

        # alpha too small
        if min_alpha is not None and alpha < min_alpha:
            return min_alpha

    # fail
    return None

class Backtracking(LineSearch):
    def __init__(
        self,
        initial_step_size: float = 1.0,
        beta: float = 0.5,
        c: float = 1e-4,
        max_iter: int = 10,
        min_alpha: float | None = None,
        adaptive=True,
    ):
        defaults=dict(initial_step_size=initial_step_size,beta=beta,c=c,max_iter=max_iter,min_alpha=min_alpha,adaptive=adaptive)
        super().__init__(defaults=defaults)
        self.global_state['beta_scale'] = 1.0

    @torch.no_grad
    def search(self, update, vars):
        initial_step_size, beta, c, max_iter, min_alpha, adaptive = itemgetter(
            'initial_step_size', 'beta', 'c', 'max_iter', 'min_alpha', 'adaptive')(self.settings[vars.params[0]])

        objective = self.make_objective(vars=vars)

        # # directional derivative
        d = -sum(t.sum() for t in torch._foreach_mul(vars.get_grad(), vars.get_update()))

        # scale beta (beta is multiplicative and i think may be better than scaling initial step size)
        if adaptive: beta = beta * self.global_state['beta_scale']

        step_size = backtracking_line_search(objective, d, initial_step_size=initial_step_size,beta=beta,
                                        c=c,max_iter=max_iter,min_alpha=min_alpha)

        # found an alpha that reduces loss
        if step_size is not None:
            self.global_state['beta_scale'] = min(1.0, self.global_state['beta_scale'] * math.sqrt(1.5))
            return step_size

        # on fail reduce beta scale value
        self.global_state['beta_scale'] /= 1.5
        return 0

def _lerp(start,end,weight):
    return start + weight * (end - start)

class AdaptiveBacktracking(LineSearch):
    def __init__(
        self,
        initial_step_size: float = 1.0,
        beta: float = 0.5,
        c: float = 1e-4,
        max_iter: int = 20,
        min_alpha: float | None = None,
        target_iters = 1,
        nplus = 2.0,
        scale_beta = 0.9,
    ):
        defaults=dict(initial_step_size=initial_step_size,beta=beta,c=c,max_iter=max_iter,min_alpha=min_alpha,target_iters=target_iters,nplus=nplus,scale_beta=scale_beta)
        super().__init__(defaults=defaults)

        self.global_state['beta_scale'] = 1.0
        self.global_state['initial_scale'] = 1.0

    @torch.no_grad
    def search(self, update, vars):
        initial_step_size, beta, c, max_iter, min_alpha, target_iters, nplus, scale_beta=itemgetter(
            'initial_step_size','beta','c','max_iter','min_alpha','target_iters','nplus','scale_beta')(self.settings[vars.params[0]])

        objective = self.make_objective(vars=vars)

        # directional derivative (0 if c = 0 because it is not needed)
        if c == 0: d = 0
        else: d = -sum(t.sum() for t in torch._foreach_mul(vars.get_grad(), update))

        # scale beta
        beta = beta * self.global_state['beta_scale']

        # scale step size so that decrease is expected at target_iters
        initial_step_size = initial_step_size * self.global_state['initial_scale']

        step_size = backtracking_line_search(objective, d, initial_step_size=initial_step_size, beta=beta,
                                        c=c,max_iter=max_iter,min_alpha=min_alpha)

        # found an alpha that reduces loss
        if step_size is not None:

            # update initial_scale
            # initial step size satisfied conditions, increase initial_scale by nplus
            if step_size == initial_step_size and target_iters > 0:
                self.global_state['initial_scale'] *= nplus ** target_iters
                self.global_state['initial_scale'] = min(self.global_state['initial_scale'], 1e32) # avoid overflow error

            else:
                # otherwise make initial_scale such that target_iters iterations will satisfy armijo
                target_intial_step_size = step_size
                for _ in range(target_iters):
                    target_intial_step_size = step_size / beta

                self.global_state['initial_scale'] = _lerp(
                    self.global_state['initial_scale'], target_intial_step_size / initial_step_size, 1-scale_beta
                )

            # revert beta_scale
            self.global_state['beta_scale'] = min(1.0, self.global_state['beta_scale'] * math.sqrt(1.5))

            return step_size

        # on fail reduce beta scale value
        self.global_state['beta_scale'] /= 1.5
        return 0
