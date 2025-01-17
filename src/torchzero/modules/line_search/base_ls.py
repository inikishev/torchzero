from typing import Literal
from abc import ABC, abstractmethod


import torch

from ... import tl
from ...core import _ClosureType, OptimizationState, OptimizerModule, _maybe_pass_backward
from ...utils.python_tools import _ScalarLoss


class MaxIterReached(Exception): pass

class LineSearchBase(OptimizerModule, ABC):
    """Base linesearch class. This is an abstract class, please don't use it as the optimizer.

    When inheriting from this class the easiest way is only override `_find_best_lr`, which should
    return the final lr to use.

    Args:
        defaults (dict): dictionary with default parameters for the module.
        target (str, optional):
            determines how _update method is used in the default step method.

            "ascent" - it updates the ascent

            "grad" - it updates the gradient (and sets `.grad` attributes to updated gradient).

            "closure" - it makes a new closure that sets the updated ascent to the .`grad` attributes.
        maxiter (_type_, optional): maximum line search iterations
            (useful for things like scipy.optimize.minimize_scalar) as it doesn't have
            an exact iteration limit. Defaults to None.
        log_lrs (bool, optional):
            saves lrs and losses with them into optimizer._lrs (for debugging).
            Defaults to False.
    """
    def __init__(
        self,
        defaults: dict,
        target: Literal['grad', 'ascent', 'closure'] = 'ascent',
        maxiter=None,
        log_lrs=False,
    ):
        super().__init__(defaults, target=target)
        self._reset()

        self.maxiter = maxiter
        self.log_lrs = log_lrs
        self._lrs: list[dict[float, _ScalarLoss]] = []
        """this only gets filled if `log_lrs` is True. On each step, a dictionary is added to this list,
        with all lrs tested at that step as keys and corresponding losses as values."""

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
    def _evaluate_lr_(self, lr: float, closure: _ClosureType, ascent: tl.TensorList, params: tl.TensorList, backward=False):
        """Evaluate `lr`, if loss is better than current lowest loss,
        overrides `self._lowest_loss` and `self._best_lr`.

        Args:
            closure (ClosureType): closure.
            params (tl.TensorList): params.
            ascent_direction (tl.TensorList): ascent.
            lr (float): lr to evaluate.

        Returns:
            Loss with evaluated lr.
        """
        # check max iter
        if self._current_iter == self.maxiter: raise MaxIterReached
        self._current_iter += 1

        # set new lr and evaluate loss with it
        self._set_lr_(lr, ascent, params = params)
        with torch.enable_grad() if backward else torch.no_grad(): self._fx0_approx = _maybe_pass_backward(closure, backward)

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
        closure: _ClosureType,
        ascent: tl.TensorList,
        params: tl.TensorList,
    ) -> float:
        """Same as _evaluate_lr_ but ensures that the loss value is float."""
        v = self._evaluate_lr_(lr, closure, ascent, params)
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
        if state.fx0_approx is None: state.fx0_approx = self._lowest_loss
        return self.next_module.step(state)

