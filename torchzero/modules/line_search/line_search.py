from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partial
from operator import itemgetter
from typing import Any

import torch

from ...core import Module, Target, Vars
from ...utils import tofloat


class MaxLineSearchItersReached(Exception): pass


class LineSearch(Module, ABC):
    """Base class for line searches.
    This is an abstract class, to use it, subclass it and override `search`.

    Args:
        defaults (dict[str, Any] | None): dictionary with defaults.
        maxiter (int | None, optional):
            if this is specified, the search method will terminate upon evaluating
            the objective this many times, and step size with the lowest loss value will be used.
            This is useful when passing `make_objective` to an external library which
            doesn't have a maxiter option. Defaults to None.
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
        step_size = tofloat(step_size)
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
        alpha = [self._current_step_size - s for s in step_size]
        if any(a!=0 for a in alpha):
            torch._foreach_add_(params, torch._foreach_mul(update, alpha))

    def _loss(self, step_size: float, vars: Vars, closure, params: list[torch.Tensor],
              update: list[torch.Tensor], backward:bool=False) -> float:

        # if step_size is 0, we might already know the loss
        if (vars.loss is not None) and (step_size == 0):
            return tofloat(vars.loss)

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

        # if evaluated loss at step size 0, set it to vars.loss
        if step_size == 0:
            vars.loss = loss

        return tofloat(loss)

    def _loss_derivative(self, step_size: float, vars: Vars, closure,
                         params: list[torch.Tensor], update: list[torch.Tensor]):
        # if step_size is 0, we might already know the derivative
        if (vars.grad is not None) and (step_size == 0):
            loss = self._loss(step_size=step_size,vars=vars,closure=closure,params=params,update=update,backward=False)
            derivative = - sum(t.sum() for t in torch._foreach_mul(vars.grad, update))

        else:
            # loss with a backward pass sets params.grad
            loss = self._loss(step_size=step_size,vars=vars,closure=closure,params=params,update=update,backward=True)

            # directional derivative
            derivative = - sum(t.sum() for t in torch._foreach_mul([p.grad if p.grad is not None
                                                                    else torch.zeros_like(p) for p in params], update))

        return loss, tofloat(derivative)

    def evaluate_step_size(self, step_size: float, vars: Vars, backward:bool=False):
        closure = vars.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return self._loss(step_size=step_size, vars=vars, closure=closure, params=vars.params,update=vars.get_update(),backward=backward)

    def evaluate_step_size_loss_and_derivative(self, step_size: float, vars: Vars):
        closure = vars.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return self._loss_derivative(step_size=step_size, vars=vars, closure=closure, params=vars.params,update=vars.get_update())

    def make_objective(self, vars: Vars, backward:bool=False):
        closure = vars.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return partial(self._loss, vars=vars, closure=closure, params=vars.params, update=vars.get_update(), backward=backward)

    def make_objective_with_derivative(self, vars: Vars):
        closure = vars.closure
        if closure is None: raise RuntimeError('line search requires closure')
        return partial(self._loss_derivative, vars=vars, closure=closure, params=vars.params, update=vars.get_update())

    @abstractmethod
    def search(self, update: list[torch.Tensor], vars: Vars) -> float:
        """Finds the step size to use"""

    @torch.no_grad
    def step(self, vars: Vars) -> Vars:
        self._reset()
        params = vars.params
        update = vars.get_update()

        try:
            step_size = self.search(update=update, vars=vars)
        except MaxLineSearchItersReached:
            step_size = self._best_step_size

        # set loss_approx
        if vars.loss_approx is None: vars.loss_approx = self._lowest_loss

        # this is last module - set step size to found step_size times lr
        if vars.is_last:

            if vars.last_module_lrs is None:
                self.set_step_size_(step_size, params=params, update=update)

            else:
                self._set_per_parameter_step_size_([step_size*lr for lr in vars.last_module_lrs], params=params, update=update)

            vars.update = None; vars.stop = True
            return vars

        # revert parameters and multiply update by step size
        self.set_step_size_(0, params=params, update=update)
        torch._foreach_mul_(vars.update, step_size)
        return vars


class GridLineSearch(LineSearch):
    def __init__(self, start, end, num):
        defaults = dict(start=start,end=end,num=num)
        super().__init__(defaults)

    @torch.no_grad
    def search(self, update, vars):
        start,end,num=itemgetter('start','end','num')(self.settings[vars.params[0]])

        for lr in torch.linspace(start,end,num):
            self.evaluate_step_size(lr.item(), vars=vars, backward=False)

        return self._best_step_size