import typing as T
from collections import abc

import numpy as np
import scipy.optimize
import torch

from ... import tl
from ...core import ClosureType, OptimizationState, OptimizerModule
from .ls_base import LineSearchBase, MaxIterReached


class GridLS(LineSearchBase):
    def __init__(
        self,
        lrs: abc.Sequence[float] | np.ndarray | torch.Tensor,
        stop_on_improvement=False,
        stop_on_worsened=False,
        log_lrs = False
    ):
        super().__init__({}, make_closure=False, maxiter=None, log_lrs=log_lrs)
        self.lrs = lrs
        self.stop_on_improvement = stop_on_improvement
        self.stop_on_worsened = stop_on_worsened

    @torch.no_grad
    def _find_best_lr(self, state: OptimizationState, params: tl.TensorList) -> float:
        if state.closure is None: raise ValueError("closure is not set")
        if state.ascent is None: raise ValueError("ascent_direction is not set")

        if self.stop_on_improvement:
            if state.fx0 is None: state.fx0 = state.closure(False)
            self._lowest_loss = state.fx0

        for lr in self.lrs:
            loss = self._evaluate_lr_(float(lr), state.closure, state.ascent, params)

            # if worsened
            if self.stop_on_worsened and loss != self._lowest_loss:
                break

            # if improved
            if self.stop_on_improvement and loss == self._lowest_loss:
                break

        return float(self._best_lr)



class MultiplicativeLS(GridLS):
    def __init__(
        self,
        init: float = 0.001,
        mul: float = 2,
        num=10,
        stop_on_improvement=False,
        stop_on_worsened=True,
    ):
        super().__init__(
            [init * mul**i for i in range(num)],
            stop_on_improvement=stop_on_improvement,
            stop_on_worsened=stop_on_worsened,
        )

class BacktrackingLS(GridLS):
    def __init__(
        self,
        init: float = 1,
        mul: float = 0.5,
        num=10,
        stop_on_improvement=True,
        stop_on_worsened=False,
    ):
        super().__init__(
            [init * mul**i for i in range(num)],
            stop_on_improvement=stop_on_improvement,
            stop_on_worsened=stop_on_worsened,
        )

class LinspaceLS(GridLS):
    def __init__(
        self,
        start: float = 0.001,
        end: float = 2,
        steps=10,
        stop_on_improvement=False,
        stop_on_worsened=False,
    ):
        super().__init__(
            torch.linspace(start, end, steps),
            stop_on_improvement=stop_on_improvement,
            stop_on_worsened=stop_on_worsened,
        )

class ArangeLS(GridLS):
    def __init__(
        self,
        start: float = 0.001,
        end: float = 2,
        step=0.1,
        stop_on_improvement=False,
        stop_on_worsened=False,
    ):
        super().__init__(
            torch.arange(start, end, step),
            stop_on_improvement=stop_on_improvement,
            stop_on_worsened=stop_on_worsened,
        )