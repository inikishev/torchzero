import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partial
from operator import itemgetter
from typing import Any, Literal

import numpy as np
import torch

from ...core import Module,  Objective
from ...utils import tofloat, set_storage_
from ..opt_utils import clip_by_finfo


class MaxLineSearchItersReached(Exception): pass


class LineSearchBase(Module, ABC):
    """Base class for line searches.

    This is an abstract class, to use it, subclass it and override `search`.

    Args:
        defaults (dict[str, Any] | None): dictionary with defaults.
        maxiter (int | None, optional):
            if this is specified, the search method will terminate upon evaluating
            the objective this many times, and step size with the lowest loss value will be used.
            This is useful when passing `make_objective` to an external library which
            doesn't have a maxiter option. Defaults to None.

    Other useful methods:
        * ``evaluate_f`` - returns loss with a given scalar step size
        * ``evaluate_f_d`` - returns loss and directional derivative with a given scalar step size
        * ``make_objective`` - creates a function that accepts a scalar step size and returns loss. This can be passed to a scalar solver, such as scipy.optimize.minimize_scalar.
        * ``make_objective_with_derivative`` - creates a function that accepts a scalar step size and returns a tuple with loss and directional derivative. This can be passed to a scalar solver.

    Examples:

    #### Basic line search

    This evaluates all step sizes in a range by using the :code:`self.evaluate_step_size` method.
    ```python
    class GridLineSearch(LineSearch):
        def __init__(self, start, end, num):
            defaults = dict(start=start,end=end,num=num)
            super().__init__(defaults)

        @torch.no_grad
        def search(self, update, var):

            start = self.defaults["start"]
            end = self.defaults["end"]
            num = self.defaults["num"]

            lowest_loss = float("inf")
            best_step_size = best_step_size

            for step_size in torch.linspace(start,end,num):
                loss = self.evaluate_step_size(step_size.item(), var=var, backward=False)
                if loss < lowest_loss:
                    lowest_loss = loss
                    best_step_size = step_size

            return best_step_size
    ```

    #### Using external solver via self.make_objective

    Here we let :code:`scipy.optimize.minimize_scalar` solver find the best step size via :code:`self.make_objective`

    ```python
    class ScipyMinimizeScalar(LineSearch):
        def __init__(self, method: str | None = None):
            defaults = dict(method=method)
            super().__init__(defaults)

        @torch.no_grad
        def search(self, update, var):
            objective = self.make_objective(var=var)
            method = self.defaults["method"]

            res = self.scopt.minimize_scalar(objective, method=method)
            return res.x
    ```
    """
    def __init__(self, defaults: dict[str, Any] | None, maxiter: int | None = None):
        super().__init__(defaults)
        self._maxiter = maxiter
        self._reset()

    def _reset(self):
        self._current_step_size: float = 0
        self._lowest_loss = float('inf')
        self._best_step_size: float = 0
        self._current_iter = 0
        self._initial_params = None

    def set_step_size_(
        self,
        step_size: float,
        params: list[torch.Tensor],
        update: list[torch.Tensor],
    ):
        if not math.isfinite(step_size): return

         # avoid overflow error
        step_size = clip_by_finfo(tofloat(step_size), torch.finfo(update[0].dtype))

        # skip is parameters are already at suggested step size
        if self._current_step_size == step_size: return

        assert self._initial_params is not None
        if step_size == 0:
            new_params = [p.clone() for p in self._initial_params]
        else:
            new_params = torch._foreach_sub(self._initial_params, update, alpha=step_size)

        for c, n in zip(params, new_params):
            set_storage_(c, n)

        self._current_step_size = step_size

    def _set_per_parameter_step_size_(
        self,
        step_size: Sequence[float],
        params: list[torch.Tensor],
        update: list[torch.Tensor],
    ):

        assert self._initial_params is not None
        if not np.isfinite(step_size).all(): step_size = [0 for _ in step_size]

        if any(s!=0 for s in step_size):
            new_params = torch._foreach_sub(self._initial_params, torch._foreach_mul(update, step_size))
        else:
            new_params = [p.clone() for p in self._initial_params]

        for c, n in zip(params, new_params):
            set_storage_(c, n)

    def _loss(self, step_size: float, var: Objective, closure, params: list[torch.Tensor],
              update: list[torch.Tensor], backward:bool=False) -> float:

        # if step_size is 0, we might already know the loss
        if (var.loss is not None) and (step_size == 0):
            return tofloat(var.loss)

        # check max iter
        if self._maxiter is not None and self._current_iter >= self._maxiter: raise MaxLineSearchItersReached
        self._current_iter += 1

        # set new lr and evaluate loss with it
        self.set_step_size_(step_size, params=params, update=update)
        if backward:
            with torch.enable_grad(): loss = closure()
        else:
            loss = closure(False)

        # if it is the best so far, record it
        if loss < self._lowest_loss:
            self._lowest_loss = tofloat(loss)
            self._best_step_size = step_size

        # if evaluated loss at step size 0, set it to var.loss
        if step_size == 0:
            var.loss = loss
            if backward: var.grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

        return tofloat(loss)

    def _loss_derivative_gradient(self, step_size: float, var: Objective, closure,
                         params: list[torch.Tensor], update: list[torch.Tensor]):
        # if step_size is 0, we might already know the derivative
        if (var.grads is not None) and (step_size == 0):
            loss = self._loss(step_size=step_size,var=var,closure=closure,params=params,update=update,backward=False)
            derivative = - sum(t.sum() for t in torch._foreach_mul(var.grads, update))

        else:
            # loss with a backward pass sets params.grad
            loss = self._loss(step_size=step_size,var=var,closure=closure,params=params,update=update,backward=True)

            # directional derivative
            derivative = - sum(t.sum() for t in torch._foreach_mul([p.grad if p.grad is not None
                                                                    else torch.zeros_like(p) for p in params], update))

        assert var.grads is not None
        return loss, tofloat(derivative), var.grads

    def _loss_derivative(self, step_size: float, var: Objective, closure,
                         params: list[torch.Tensor], update: list[torch.Tensor]):
        return self._loss_derivative_gradient(step_size=step_size, var=var,closure=closure,params=params,update=update)[:2]

    def evaluate_f(self, step_size: float, var: Objective, backward:bool=False):
        """evaluate function value at alpha `step_size`."""
        closure = var.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return self._loss(step_size=step_size, var=var, closure=closure, params=var.params,update=var.get_updates(),backward=backward)

    def evaluate_f_d(self, step_size: float, var: Objective):
        """evaluate function value and directional derivative in the direction of the update at step size `step_size`."""
        closure = var.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return self._loss_derivative(step_size=step_size, var=var, closure=closure, params=var.params,update=var.get_updates())

    def evaluate_f_d_g(self, step_size: float, var: Objective):
        """evaluate function value, directional derivative, and gradient list at step size `step_size`."""
        closure = var.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return self._loss_derivative_gradient(step_size=step_size, var=var, closure=closure, params=var.params,update=var.get_updates())

    def make_objective(self, var: Objective, backward:bool=False):
        closure = var.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return partial(self._loss, var=var, closure=closure, params=var.params, update=var.get_updates(), backward=backward)

    def make_objective_with_derivative(self, var: Objective):
        closure = var.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return partial(self._loss_derivative, var=var, closure=closure, params=var.params, update=var.get_updates())

    def make_objective_with_derivative_and_gradient(self, var: Objective):
        closure = var.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return partial(self._loss_derivative_gradient, var=var, closure=closure, params=var.params, update=var.get_updates())

    @abstractmethod
    def search(self, update: list[torch.Tensor], var: Objective) -> float:
        """Finds the step size to use"""

    @torch.no_grad
    def apply(self, objective: Objective) -> Objective:
        self._reset()

        params = objective.params
        self._initial_params = [p.clone() for p in params]
        update = objective.get_updates()

        try:
            step_size = self.search(update=update, var=objective)
        except MaxLineSearchItersReached:
            step_size = self._best_step_size

        step_size = clip_by_finfo(step_size, torch.finfo(update[0].dtype))

        # set loss_approx
        if objective.loss_approx is None: objective.loss_approx = self._lowest_loss

        # if this is last module, directly update parameters to avoid redundant operations
        if objective.modular is not None and self is objective.modular.modules[-1]:
            self.set_step_size_(step_size, params=params, update=update)

            objective.stop = True; objective.skip_update = True
            return objective

        # revert parameters and multiply update by step size
        self.set_step_size_(0, params=params, update=update)
        torch._foreach_mul_(objective.updates, step_size)
        return objective



class GridLineSearch(LineSearchBase):
    """"""
    def __init__(self, start, end, num):
        defaults = dict(start=start,end=end,num=num)
        super().__init__(defaults)

    @torch.no_grad
    def search(self, update, var):
        start, end, num = itemgetter('start', 'end', 'num')(self.defaults)

        for lr in torch.linspace(start,end,num):
            self.evaluate_f(lr.item(), var=var, backward=False)

        return self._best_step_size


def sufficient_decrease(f_0, g_0, f_a, a, c):
    return f_a < f_0 + c*a*min(g_0, 0)

def curvature(g_0, g_a, c):
    if g_0 > 0: return True
    return g_a >= c * g_0

def strong_curvature(g_0, g_a, c):
    """same as curvature condition except curvature can't be too positive (which indicates overstep)"""
    if g_0 > 0: return True
    return abs(g_a) <= c * abs(g_0)

def wolfe(f_0, g_0, f_a, g_a, a, c1, c2):
    return sufficient_decrease(f_0, g_0, f_a, a, c1) and curvature(g_0, g_a, c2)

def strong_wolfe(f_0, g_0, f_a, g_a, a, c1, c2):
    return sufficient_decrease(f_0, g_0, f_a, a, c1) and strong_curvature(g_0, g_a, c2)

def goldstein(f_0, g_0, f_a, a, c):
    """same as armijo (sufficient_decrease) but additional lower bound"""
    g_0 = min(g_0, 0)
    return f_0 + (1-c)*a*g_0 < f_a < f_0 + c*a*g_0

TerminationCondition = Literal["armijo", "curvature", "strong_curvature", "wolfe", "strong_wolfe", "goldstein", "decrease"]
def termination_condition(
    condition: TerminationCondition,
    f_0,
    g_0,
    f_a,
    g_a: Any | None,
    a,
    c,
    c2=None,
):
    if not math.isfinite(f_a): return False
    if condition == 'armijo': return sufficient_decrease(f_0, g_0, f_a, a, c)
    if condition == 'curvature': return curvature(g_0, g_a, c)
    if condition == 'strong_curvature': return strong_curvature(g_0, g_a, c)
    if condition == 'wolfe': return wolfe(f_0, g_0, f_a, g_a, a, c, c2)
    if condition == 'strong_wolfe': return strong_wolfe(f_0, g_0, f_a, g_a, a, c, c2)
    if condition == 'goldstein': return goldstein(f_0, g_0, f_a, a, c)
    if condition == 'decrease': return f_a < f_0
    raise ValueError(f"unknown condition {condition}")