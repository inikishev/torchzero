import math
from collections.abc import Callable
from operator import itemgetter

import torch

from .line_search import LineSearchBase


def backtracking_line_search(
    f: Callable[[float], float],
    g_0: float | torch.Tensor,
    init: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
    maxiter: int = 10,
    try_negative: bool = False,
) -> float | None:
    """

    Args:
        f: evaluates step size along some descent direction.
        g_0: directional derivative along the descent direction.
        init: initial step size.
        beta: The factor by which to decrease alpha in each iteration
        c: The constant for the Armijo sufficient decrease condition
        maxiter: Maximum number of backtracking iterations (default: 10).

    Returns:
        step size
    """

    a = init
    f_x = f(0)
    f_prev = None

    for iteration in range(maxiter):
        f_a = f(a)

        if (f_prev is not None) and (f_a > f_prev) and (f_prev < f_x): return a / beta
        f_prev = f_a

        if f_a < f_x + c * a * min(g_0, 0): # pyright: ignore[reportArgumentType]
            # found an acceptable alpha
            return a

        # decrease alpha
        a *= beta

    # fail
    if try_negative:
        def inv_objective(alpha): return f(-alpha)

        v = backtracking_line_search(
            inv_objective,
            g_0=-g_0,
            beta=beta,
            c=c,
            maxiter=maxiter,
            try_negative=False,
        )
        if v is not None: return -v

    return None

class Backtracking(LineSearchBase):
    """Backtracking line search satisfying the Armijo condition.

    Args:
        init (float, optional): initial step size. Defaults to 1.0.
        beta (float, optional): multiplies each consecutive step size by this value. Defaults to 0.5.
        c (float, optional): acceptance value for Armijo condition. Defaults to 1e-4.
        maxiter (int, optional): Maximum line search function evaluations. Defaults to 10.
        adaptive (bool, optional):
            when enabled, if line search failed, beta is reduced.
            Otherwise it is reset to initial value. Defaults to True.
        try_negative (bool, optional): Whether to perform line search in opposite direction on fail. Defaults to False.

    Examples:
        Gradient descent with backtracking line search:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.Backtracking()
            )

        LBFGS with backtracking line search:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.LBFGS(),
                tz.m.Backtracking()
            )

    """
    def __init__(
        self,
        init: float = 1.0,
        beta: float = 0.5,
        c: float = 1e-4,
        maxiter: int = 10,
        adaptive=True,
        try_negative: bool = False,
    ):
        defaults=dict(init=init,beta=beta,c=c,maxiter=maxiter,adaptive=adaptive, try_negative=try_negative)
        super().__init__(defaults=defaults)
        self.global_state['beta_scale'] = 1.0

    def reset(self):
        super().reset()
        self.global_state['beta_scale'] = 1.0

    @torch.no_grad
    def search(self, update, var):
        init, beta, c, maxiter, adaptive, try_negative = itemgetter(
            'init', 'beta', 'c', 'maxiter', 'adaptive', 'try_negative')(self.settings[var.params[0]])

        objective = self.make_objective(var=var)

        # # directional derivative
        d = -sum(t.sum() for t in torch._foreach_mul(var.get_grad(), var.get_update()))

        # scale beta (beta is multiplicative and i think may be better than scaling initial step size)
        if adaptive: beta = beta * self.global_state['beta_scale']

        step_size = backtracking_line_search(objective, d, init=init,beta=beta,
                                        c=c,maxiter=maxiter, try_negative=try_negative)

        # found an alpha that reduces loss
        if step_size is not None:
            self.global_state['beta_scale'] = min(1.0, self.global_state['beta_scale'] * math.sqrt(1.5))
            return step_size

        # on fail reduce beta scale value
        self.global_state['beta_scale'] /= 1.5
        return 0

def _lerp(start,end,weight):
    return start + weight * (end - start)

class AdaptiveBacktracking(LineSearchBase):
    """Adaptive backtracking line search. After each line search procedure, a new initial step size is set
    such that optimal step size in the procedure would be found on the second line search iteration.

    Args:
        init (float, optional): step size for the first step. Defaults to 1.0.
        beta (float, optional): multiplies each consecutive step size by this value. Defaults to 0.5.
        c (float, optional): acceptance value for Armijo condition. Defaults to 1e-4.
        maxiter (int, optional): Maximum line search function evaluations. Defaults to 10.
        target_iters (int, optional):
            target number of iterations that would be performed until optimal step size is found. Defaults to 1.
        nplus (float, optional):
            Multiplier to initial step size if it was found to be the optimal step size. Defaults to 2.0.
        scale_beta (float, optional):
            Momentum for initial step size, at 0 disables momentum. Defaults to 0.0.
        try_negative (bool, optional): Whether to perform line search in opposite direction on fail. Defaults to False.
    """
    def __init__(
        self,
        init: float = 1.0,
        beta: float = 0.5,
        c: float = 1e-4,
        maxiter: int = 20,
        target_iters = 1,
        nplus = 2.0,
        scale_beta = 0.0,
        try_negative: bool = False,
    ):
        defaults=dict(init=init,beta=beta,c=c,maxiter=maxiter,target_iters=target_iters,nplus=nplus,scale_beta=scale_beta, try_negative=try_negative)
        super().__init__(defaults=defaults)

        self.global_state['beta_scale'] = 1.0
        self.global_state['initial_scale'] = 1.0

    def reset(self):
        super().reset()
        self.global_state['beta_scale'] = 1.0
        self.global_state['initial_scale'] = 1.0

    @torch.no_grad
    def search(self, update, var):
        init, beta, c, maxiter, target_iters, nplus, scale_beta, try_negative=itemgetter(
            'init','beta','c','maxiter','target_iters','nplus','scale_beta', 'try_negative')(self.settings[var.params[0]])

        objective = self.make_objective(var=var)

        # directional derivative (0 if c = 0 because it is not needed)
        if c == 0: d = 0
        else: d = -sum(t.sum() for t in torch._foreach_mul(var.get_grad(), update))

        # scale beta
        beta = beta * self.global_state['beta_scale']

        # scale step size so that decrease is expected at target_iters
        init = init * self.global_state['initial_scale']

        step_size = backtracking_line_search(objective, d, init=init, beta=beta,
                                        c=c,maxiter=maxiter, try_negative=try_negative)

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
