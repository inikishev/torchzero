import typing

import torch
try:
    import scipy.optimize as scopt
except ModuleNotFoundError:
    scopt = typing.cast(typing.Any, None)

from ... import tl
from ...core import OptimizationState

from .ls_base import LineSearchBase, MaxIterReached

if typing.TYPE_CHECKING:
    import scipy.optimize as scopt

class ScipyMinimizeScalarLS(LineSearchBase):
    """Line search via `scipy.optimize.minimize_scalar`. All args except maxiter are the same as for it.

    Args:
        method (Optional[str], optional): 'brent', 'golden' or 'bounded'. Defaults to None.
        maxiter (Optional[int], optional): hard limit on maximum number of function evaluations. Defaults to None.
        bracket (optional): bracket. Defaults to None.
        bounds (optional): bounds. Defaults to None.
        tol (Optional[float], optional): some kind of tolerance. Defaults to None.
        options (optional): options for method. Defaults to None.
        log_lrs (bool, optional): logs lrs and values into `_lrs`. Defaults to False.
    """
    def __init__(
        self,
        method: str | None = None,
        maxiter: int | None = None,
        bracket = None,
        bounds = None,
        tol: float | None = None,
        options = None,
        log_lrs = False,
    ):
        if scopt is None: raise ModuleNotFoundError("scipy is not installed")
        super().__init__({}, make_closure=False, maxiter=maxiter, log_lrs=log_lrs)
        self.method = method
        self.tol = tol
        self.bracket = bracket
        self.bounds = bounds
        self.options = options

    @torch.no_grad
    def _find_best_lr(self, state: OptimizationState, params: tl.TensorList) -> float:
        try:
            res = scopt.minimize_scalar(
                self._evaluate_lr_ensure_float,
                args = (state.closure, state.ascent, params),
                method = self.method,
                tol = self.tol,
                bracket = self.bracket,
                bounds = self.bounds,
                options = self.options,
            ) # type:ignore
        except MaxIterReached:
            pass

        return float(self._best_lr)