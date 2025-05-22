import math
from collections.abc import Callable
from operator import itemgetter

import torch

from .line_search import LineSearch


def backtracking_line_search(
    f: Callable[[float], float],
    g_0: float | torch.Tensor,
    init: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
    maxiter: int = 10,
    a_min: float | None = None,
    try_negative: bool = False,
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

    a = init
    f_x = f(0)

    for iteration in range(maxiter):
        f_a = f(a)

        if f_a <= f_x + c * a * min(g_0, 0): # pyright: ignore[reportArgumentType]
            # found an acceptable alpha
            return a

        # decrease alpha
        a *= beta

        # alpha too small
        if a_min is not None and a < a_min:
            return a_min

    # fail
    if try_negative:
        def inv_objective(alpha): return f(-alpha)

        v = backtracking_line_search(
            inv_objective,
            g_0=-g_0,
            beta=beta,
            c=c,
            maxiter=maxiter,
            a_min=a_min,
            try_negative=False,
        )
        if v is not None: return -v

    return None

class Backtracking(LineSearch):
    def __init__(
        self,
        init: float = 1.0,
        beta: float = 0.5,
        c: float = 1e-4,
        maxiter: int = 10,
        min_alpha: float | None = None,
        adaptive=True,
        try_negative: bool = False,
    ):
        defaults=dict(init=init,beta=beta,c=c,maxiter=maxiter,min_alpha=min_alpha,adaptive=adaptive, try_negative=try_negative)
        super().__init__(defaults=defaults)
        self.global_state['beta_scale'] = 1.0

    def reset(self):
        super().reset()
        self.global_state['beta_scale'] = 1.0

    @torch.no_grad
    def search(self, update, vars):
        init, beta, c, maxiter, min_alpha, adaptive, try_negative = itemgetter(
            'init', 'beta', 'c', 'maxiter', 'min_alpha', 'adaptive', 'try_negative')(self.settings[vars.params[0]])

        objective = self.make_objective(vars=vars)

        # # directional derivative
        d = -sum(t.sum() for t in torch._foreach_mul(vars.get_grad(), vars.get_update()))

        # scale beta (beta is multiplicative and i think may be better than scaling initial step size)
        if adaptive: beta = beta * self.global_state['beta_scale']

        step_size = backtracking_line_search(objective, d, init=init,beta=beta,
                                        c=c,maxiter=maxiter,a_min=min_alpha, try_negative=try_negative)

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
        init: float = 1.0,
        beta: float = 0.5,
        c: float = 1e-4,
        maxiter: int = 20,
        min_alpha: float | None = None,
        target_iters = 1,
        nplus = 2.0,
        scale_beta = 0.0,
        try_negative: bool = False,
    ):
        defaults=dict(init=init,beta=beta,c=c,maxiter=maxiter,min_alpha=min_alpha,target_iters=target_iters,nplus=nplus,scale_beta=scale_beta, try_negative=try_negative)
        super().__init__(defaults=defaults)

        self.global_state['beta_scale'] = 1.0
        self.global_state['initial_scale'] = 1.0

    def reset(self):
        super().reset()
        self.global_state['beta_scale'] = 1.0
        self.global_state['initial_scale'] = 1.0

    @torch.no_grad
    def search(self, update, vars):
        init, beta, c, maxiter, min_alpha, target_iters, nplus, scale_beta, try_negative=itemgetter(
            'init','beta','c','maxiter','min_alpha','target_iters','nplus','scale_beta', 'try_negative')(self.settings[vars.params[0]])

        objective = self.make_objective(vars=vars)

        # directional derivative (0 if c = 0 because it is not needed)
        if c == 0: d = 0
        else: d = -sum(t.sum() for t in torch._foreach_mul(vars.get_grad(), update))

        # scale beta
        beta = beta * self.global_state['beta_scale']

        # scale step size so that decrease is expected at target_iters
        init = init * self.global_state['initial_scale']

        step_size = backtracking_line_search(objective, d, init=init, beta=beta,
                                        c=c,maxiter=maxiter,a_min=min_alpha, try_negative=try_negative)

        # found an alpha that reduces loss
        if step_size is not None:

            # update initial_scale
            # initial step size satisfied conditions, increase initial_scale by nplus
            if step_size == init and target_iters > 0:
                self.global_state['initial_scale'] *= nplus ** target_iters
                self.global_state['initial_scale'] = min(self.global_state['initial_scale'], 1e32) # avoid overflow error

            else:
                # otherwise make initial_scale such that target_iters iterations will satisfy armijo
                init_target = step_size
                for _ in range(target_iters):
                    init_target = step_size / beta

                self.global_state['initial_scale'] = _lerp(
                    self.global_state['initial_scale'], init_target / init, 1-scale_beta
                )

            # revert beta_scale
            self.global_state['beta_scale'] = min(1.0, self.global_state['beta_scale'] * math.sqrt(1.5))

            return step_size

        # on fail reduce beta scale value
        self.global_state['beta_scale'] /= 1.5
        return 0
