import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partial
from operator import itemgetter
from typing import Any

import numpy as np
import torch

from ...core import Module, Target, Var
from ...utils import tofloat


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
        * `evaluate_step_size` - returns loss with a given scalar step size
        * `evaluate_step_size_loss_and_derivative` - returns loss and directional derivative with a given scalar step size
        * `make_objective` - creates a function that accepts a scalar step size and returns loss. This can be passed to a scalar solver, such as scipy.optimize.minimize_scalar.
        * `make_objective_with_derivative` - creates a function that accepts a scalar step size and returns a tuple with loss and directional derivative. This can be passed to a scalar solver.

    Examples:
        #### Basic line search

        This evaluates all step sizes in a range by using the :code:`self.evaluate_step_size` method.

        .. code-block:: python

            class GridLineSearch(LineSearch):
                def __init__(self, start, end, num):
                    defaults = dict(start=start,end=end,num=num)
                    super().__init__(defaults)

                @torch.no_grad
                def search(self, update, var):
                    settings = self.settings[var.params[0]]
                    start = settings["start"]
                    end = settings["end"]
                    num = settings["num"]

                    lowest_loss = float("inf")
                    best_step_size = best_step_size

                    for step_size in torch.linspace(start,end,num):
                        loss = self.evaluate_step_size(step_size.item(), var=var, backward=False)
                        if loss < lowest_loss:
                            lowest_loss = loss
                            best_step_size = step_size

                    return best_step_size

        #### Using external solver via self.make_objective

        Here we let :code:`scipy.optimize.minimize_scalar` solver find the best step size via :code:`self.make_objective`

        .. code-block:: python

            class ScipyMinimizeScalar(LineSearch):
                def __init__(self, method: str | None = None):
                    defaults = dict(method=method)
                    super().__init__(defaults)

                @torch.no_grad
                def search(self, update, var):
                    objective = self.make_objective(var=var)
                    method = self.settings[var.params[0]]["method"]

                    res = self.scopt.minimize_scalar(objective, method=method)
                    return res.x

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

    def set_step_size_(
        self,
        step_size: float,
        params: list[torch.Tensor],
        update: list[torch.Tensor],
    ):
        if not math.isfinite(step_size): return
        step_size = max(min(tofloat(step_size), 1e36), -1e36) # fixes overflow when backtracking keeps increasing alpha after converging
        alpha = self._current_step_size - step_size
        if alpha != 0:
            torch._foreach_add_(params, update, alpha=alpha)
        self._current_step_size = step_size

    def _set_per_parameter_step_size_(
        self,
        step_size: Sequence[float],
        params: list[torch.Tensor],
        update: list[torch.Tensor],
    ):
        if not np.isfinite(step_size): step_size = [0 for _ in step_size]
        alpha = [self._current_step_size - s for s in step_size]
        if any(a!=0 for a in alpha):
            torch._foreach_add_(params, torch._foreach_mul(update, alpha))

    def _loss(self, step_size: float, var: Var, closure, params: list[torch.Tensor],
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
            if backward: var.grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

        return tofloat(loss)

    def _loss_derivative(self, step_size: float, var: Var, closure,
                         params: list[torch.Tensor], update: list[torch.Tensor]):
        # if step_size is 0, we might already know the derivative
        if (var.grad is not None) and (step_size == 0):
            loss = self._loss(step_size=step_size,var=var,closure=closure,params=params,update=update,backward=False)
            derivative = - sum(t.sum() for t in torch._foreach_mul(var.grad, update))

        else:
            # loss with a backward pass sets params.grad
            loss = self._loss(step_size=step_size,var=var,closure=closure,params=params,update=update,backward=True)

            # directional derivative
            derivative = - sum(t.sum() for t in torch._foreach_mul([p.grad if p.grad is not None
                                                                    else torch.zeros_like(p) for p in params], update))

        return loss, tofloat(derivative)

    def evaluate_step_size(self, step_size: float, var: Var, backward:bool=False):
        closure = var.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return self._loss(step_size=step_size, var=var, closure=closure, params=var.params,update=var.get_update(),backward=backward)

    def evaluate_step_size_loss_and_derivative(self, step_size: float, var: Var):
        closure = var.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return self._loss_derivative(step_size=step_size, var=var, closure=closure, params=var.params,update=var.get_update())

    def make_objective(self, var: Var, backward:bool=False):
        closure = var.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return partial(self._loss, var=var, closure=closure, params=var.params, update=var.get_update(), backward=backward)

    def make_objective_with_derivative(self, var: Var):
        closure = var.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return partial(self._loss_derivative, var=var, closure=closure, params=var.params, update=var.get_update())

    @abstractmethod
    def search(self, update: list[torch.Tensor], var: Var) -> float:
        """Finds the step size to use"""

    @torch.no_grad
    def step(self, var: Var) -> Var:
        self._reset()
        params = var.params
        update = var.get_update()

        try:
            step_size = self.search(update=update, var=var)
        except MaxLineSearchItersReached:
            step_size = self._best_step_size

        # set loss_approx
        if var.loss_approx is None: var.loss_approx = self._lowest_loss

        # this is last module - set step size to found step_size times lr
        if var.is_last:

            if var.last_module_lrs is None:
                self.set_step_size_(step_size, params=params, update=update)

            else:
                self._set_per_parameter_step_size_([step_size*lr for lr in var.last_module_lrs], params=params, update=update)

            var.stop = True; var.skip_update = True
            return var

        # revert parameters and multiply update by step size
        self.set_step_size_(0, params=params, update=update)
        torch._foreach_mul_(var.update, step_size)
        return var



# class GridLineSearch(LineSearch):
#     """Mostly for testing, this is not practical"""
#     def __init__(self, start, end, num):
#         defaults = dict(start=start,end=end,num=num)
#         super().__init__(defaults)

#     @torch.no_grad
#     def search(self, update, var):
#         start,end,num=itemgetter('start','end','num')(self.settings[var.params[0]])

#         for lr in torch.linspace(start,end,num):
#             self.evaluate_step_size(lr.item(), var=var, backward=False)

#         return self._best_step_size