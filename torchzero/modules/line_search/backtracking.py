import math
from collections.abc import Callable
from operator import itemgetter

import torch

from .line_search import LineSearchBase, TerminationCondition, termination_condition


def backtracking_line_search(
    f: Callable[[float], float],
    g_0: float | torch.Tensor,
    init: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
    maxiter: int = 10,
    condition: TerminationCondition = 'armijo',
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
    f_0 = f(0)
    f_prev = None

    for iteration in range(maxiter):
        f_a = f(a)
        if not math.isfinite(f_a):
            a *= beta
            continue

        if (f_prev is not None) and (f_a > f_prev) and (f_prev < f_0):
            return a / beta # new value is larger than previous value
        f_prev = f_a

        if termination_condition(condition, f_0=f_0, g_0=g_0, f_a=f_a, g_a=None, a=a, c=c):
            # found an acceptable alpha
            return a

        # decrease alpha
        a *= beta

    # fail
    return None

class Backtracking(LineSearchBase):
    """Backtracking line search.

    Args:
        init (float, optional): initial step size. Defaults to 1.0.
        beta (float, optional): multiplies each consecutive step size by this value. Defaults to 0.5.
        c (float, optional): sufficient decrease condition. Defaults to 1e-4.
        condition (TerminationCondition, optional):
            termination condition, only ones that do not use gradient at f(x+a*d) can be specified.
            - "armijo" - sufficient decrease condition.
            - "decrease" - any decrease in objective function value satisfies the condition.

            "goldstein" can techincally be specified but it doesn't make sense because there is not zoom stage.
            Defaults to 'armijo'.
        maxiter (int, optional): maximum number of function evaluations per step. Defaults to 10.
        adaptive (bool, optional):
            when enabled, if line search failed, step size will continue decreasing on the next step.
            Otherwise it will restart the line search from ``init`` step size. Defaults to True.

    Examples:
    Gradient descent with backtracking line search:

    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Backtracking()
    )
    ```

    L-BFGS with backtracking line search:
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.LBFGS(),
        tz.m.Backtracking()
    )
    ```

    """
    def __init__(
        self,
        init: float = 1.0,
        beta: float = 0.5,
        c: float = 1e-4,
        condition: TerminationCondition = 'armijo',
        maxiter: int = 10,
        adaptive=True,
    ):
        defaults=dict(init=init,beta=beta,c=c,condition=condition,maxiter=maxiter,adaptive=adaptive)
        super().__init__(defaults=defaults)

    def reset(self):
        super().reset()

    @torch.no_grad
    def search(self, update, var):
        init, beta, c, condition, maxiter, adaptive = itemgetter(
            'init', 'beta', 'c', 'condition', 'maxiter', 'adaptive')(self.defaults)

        objective = self.make_objective(var=var)

        # # directional derivative
        if c == 0: d = 0
        else: d = -sum(t.sum() for t in torch._foreach_mul(var.get_grads(), var.get_updates()))

        # scale init
        init_scale = self.global_state.get('init_scale', 1)
        if adaptive: init = init * init_scale

        step_size = backtracking_line_search(objective, d, init=init, beta=beta,c=c, condition=condition, maxiter=maxiter)

        # found an alpha that reduces loss
        if step_size is not None:
            #self.global_state['beta_scale'] = min(1.0, self.global_state['beta_scale'] * math.sqrt(1.5))
            self.global_state['init_scale'] = 1
            return step_size

        # on fail set init_scale to continue decreasing the step size
        # or set to large step size when it becomes too small
        if adaptive:
            finfo = torch.finfo(var.params[0].dtype)
            if init_scale <= finfo.tiny * 2:
                self.global_state["init_scale"] = init * 2
            else:
                self.global_state['init_scale'] = init_scale * beta**maxiter
        return 0

def _lerp(start,end,weight):
    return start + weight * (end - start)

class AdaptiveBacktracking(LineSearchBase):
    """Adaptive backtracking line search. After each line search procedure, a new initial step size is set
    such that optimal step size in the procedure would be found on the second line search iteration.

    Args:
        init (float, optional): initial step size. Defaults to 1.0.
        beta (float, optional): multiplies each consecutive step size by this value. Defaults to 0.5.
        c (float, optional): sufficient decrease condition. Defaults to 1e-4.
        condition (TerminationCondition, optional):
            termination condition, only ones that do not use gradient at f(x+a*d) can be specified.
            - "armijo" - sufficient decrease condition.
            - "decrease" - any decrease in objective function value satisfies the condition.

            "goldstein" can techincally be specified but it doesn't make sense because there is not zoom stage.
            Defaults to 'armijo'.
        maxiter (int, optional): maximum number of function evaluations per step. Defaults to 10.
        target_iters (int, optional):
            sets next step size such that this number of iterations are expected
            to be performed until optimal step size is found. Defaults to 1.
        nplus (float, optional):
            if initial step size is optimal, it is multiplied by this value. Defaults to 2.0.
        scale_beta (float, optional):
            momentum for initial step size, at 0 disables momentum. Defaults to 0.0.
    """
    def __init__(
        self,
        init: float = 1.0,
        beta: float = 0.5,
        c: float = 1e-4,
        condition: TerminationCondition = 'armijo',
        maxiter: int = 20,
        target_iters = 1,
        nplus = 2.0,
        scale_beta = 0.0,
    ):
        defaults=dict(init=init,beta=beta,c=c,condition=condition,maxiter=maxiter,target_iters=target_iters,nplus=nplus,scale_beta=scale_beta)
        super().__init__(defaults=defaults)

        self.global_state['beta_scale'] = 1.0
        self.global_state['initial_scale'] = 1.0

    def reset(self):
        super().reset()
        self.global_state['beta_scale'] = 1.0
        self.global_state['initial_scale'] = 1.0

    @torch.no_grad
    def search(self, update, var):
        init, beta, c,condition, maxiter, target_iters, nplus, scale_beta=itemgetter(
            'init','beta','c','condition', 'maxiter','target_iters','nplus','scale_beta')(self.defaults)

        objective = self.make_objective(var=var)

        # directional derivative (0 if c = 0 because it is not needed)
        if c == 0: d = 0
        else: d = -sum(t.sum() for t in torch._foreach_mul(var.get_grads(), update))

        # scale beta
        beta = beta * self.global_state['beta_scale']

        # scale step size so that decrease is expected at target_iters
        init = init * self.global_state['initial_scale']

        step_size = backtracking_line_search(objective, d, init=init, beta=beta, c=c, condition=condition, maxiter=maxiter)

        # found an alpha that reduces loss
        if step_size is not None:

            # update initial_scale
            # initial step size satisfied conditions, increase initial_scale by nplus
            if step_size == init and target_iters > 0:
                self.global_state['initial_scale'] *= nplus ** target_iters

                # clip by maximum possibel value to avoid overflow exception
                self.global_state['initial_scale'] = min(
                    self.global_state['initial_scale'],
                    torch.finfo(var.params[0].dtype).max / 2,
                )

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
