import typing as T

import torch
import scipy.optimize

from ... import tl
from ...core import OptimizerModule, OptimizationState, ClosureType

from .ls_base import LineSearchBase, MaxIterReached

class ScipyMinimizeScalarLS(LineSearchBase):
    def __init__(
        self,
        method: T.Optional[str] = None,
        maxiter: T.Optional[int] = None,
        bracket = None,
        bounds = None,
        tol: T.Optional[float] = None,
        options = None,
        log_lrs = False,
    ):
        super().__init__({}, False, maxiter=maxiter, log_lrs=log_lrs)
        self.method = method
        self.tol = tol
        self.bracket = bracket
        self.bounds = bounds
        self.options = options

    @torch.no_grad
    def _find_best_lr(self, state: OptimizationState, params: tl.TensorList) -> float:
        try:
            res: scipy.optimize.OptimizeResult = scipy.optimize.minimize_scalar(
                self._evaluate_lr_ensure_float,
                args = (state.closure, state.ascent_direction, params),
                method = self.method,
                tol = self.tol,
                bracket = self.bracket,
                bounds = self.bounds,
                options = self.options,
            ) # type:ignore
        except MaxIterReached:
            pass

        return float(self._best_lr)