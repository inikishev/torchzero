import typing as T
from abc import ABC, abstractmethod
from collections import abc
from contextlib import nullcontext

import torch

from ... import tl
from ...core import ClosureType, OptimizationState, OptimizerModule
from ...python_tools import ScalarType


class MaxIterReached(Exception): pass

class LineSearchBase(OptimizerModule, ABC):
    def __init__(
        self,
        defaults: dict,
        make_closure=False,
        maxiter=None,
        log_lrs=False,
    ):
        super().__init__(defaults, make_closure=make_closure)
        self._reset()

        self.maxiter = maxiter
        self.log_lrs = log_lrs
        self._lrs: list[dict[float, ScalarType]] = []

    def _reset(self):
        """Resets `_last_lr`, `_lowest_loss`, `_best_lr`, `_fx0_approx` and `_current_iter`."""
        self._last_lr = 0
        self._lowest_loss = float('inf')
        self._best_lr = 0
        self._fx0_approx = None
        self._current_iter = 0

    def _set_lr_(self, lr: float, ascent_direction: tl.TensorList, params: tl.TensorList, ):
        alpha = self._last_lr - lr
        if alpha != 0: params.add_(ascent_direction, alpha = alpha)
        self._last_lr = lr

    # lr is first here so that we can use a partial
    def _evaluate_lr_(self, lr: float, closure: ClosureType, ascent_direction: tl.TensorList, params: tl.TensorList, backward=False):
        """Evaluate `lr`, if loss is better than current lowest loss,
        overrides `self._lowest_loss` and `self._best_lr`.

        Args:
            closure (ClosureType): _description_
            params (tl.TensorList): _description_
            ascent_direction (tl.TensorList): _description_
            lr (float): _description_

        Returns:
            Loss with evaluated lr.
        """
        # check max iter
        if self._current_iter == self.maxiter: raise MaxIterReached
        self._current_iter += 1

        # set new lr and evaluate loss with it
        self._set_lr_(lr, ascent_direction, params = params)
        with torch.enable_grad() if backward else nullcontext(): self._fx0_approx = closure(backward)

        # if it is the best so far, record it
        if self._fx0_approx < self._lowest_loss:
            self._lowest_loss = self._fx0_approx
            self._best_lr = lr

        # log lr and loss
        if self.log_lrs:
            self._lrs[-1][lr] = self._fx0_approx

        return self._fx0_approx

    def _evaluate_lr_ensure_float(
        self,
        lr: float,
        closure: ClosureType,
        ascent_direction: tl.TensorList,
        params: tl.TensorList,
    ) -> float:
        """Same as _evaluate_lr_ but ensures that the loss value is float."""
        v = self._evaluate_lr_(lr, closure, ascent_direction, params)
        if isinstance(v, torch.Tensor): return v.detach().cpu().item()
        return float(v)

    def _find_best_lr(self, state: OptimizationState, params: tl.TensorList) -> float:
        """This should return the best lr."""
        ... # pylint:disable=unnecessary-ellipsis

    @torch.no_grad
    def step(self, state: OptimizationState):
        self._reset()
        if self.log_lrs: self._lrs.append({})

        params = self.get_params()
        ascent_direction = state.maybe_use_grad_(params)

        try:
            lr = self._find_best_lr(state, params) # pylint:disable=assignment-from-no-return
        except MaxIterReached:
            lr = self._best_lr

        # if child is None, set best lr which update params and return loss
        if self.next_module is None:
            self._set_lr_(lr, ascent_direction, params)
            return self._lowest_loss

        # otherwise undo the update by setting lr to 0 and instead multiply ascent direction by lr.
        self._set_lr_(0, ascent_direction, params)
        ascent_direction.mul_(self._best_lr)
        state.ascent = ascent_direction
        return self.next_module.step(state)

