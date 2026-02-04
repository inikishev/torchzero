import math
from collections.abc import Mapping
from operator import itemgetter

import torch

from .line_search import LineSearchBase


class ScipyMinimizeScalar(LineSearchBase):
    """Line search via :code:`scipy.optimize.minimize_scalar` which implements brent, golden search and bounded brent methods.

    Args:
        method (str | None, optional): "brent", "golden" or "bounded". Defaults to None.
        maxiter (int | None, optional): maximum number of function evaluations the line search is allowed to perform. Defaults to None.
        bracket (Sequence | None, optional):
            Either a triple (xa, xb, xc) satisfying xa < xb < xc and func(xb) < func(xa) and  func(xb) < func(xc), or a pair (xa, xb) to be used as initial points for a downhill bracket search. Defaults to None.
        bounds (Sequence | None, optional):
            For method ‘bounded’, bounds is mandatory and must have two finite items corresponding to the optimization bounds. Defaults to None.
        tol (float | None, optional): Tolerance for termination. Defaults to None.
        prev_init (bool, optional): uses previous step size as initial guess for the line search.
        options (dict | None, optional): A dictionary of solver options. Defaults to None.

    For more details on methods and arguments refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

    """
    def __init__(
        self,
        method: str | None = None,
        maxiter: int | None = None,
        bracket=None,
        bounds=None,
        tol: float | None = None,
        prev_init: bool = False,
        options=None,
    ):
        defaults = dict(method=method,bracket=bracket,bounds=bounds,tol=tol,options=options,maxiter=maxiter, prev_init=prev_init)
        super().__init__(defaults)

        import scipy.optimize
        self.scopt = scipy.optimize


    @torch.no_grad
    def search(self, update, var):
        objective = self.make_objective(var=var)
        method, bracket, bounds, tol, options, maxiter = itemgetter(
            'method', 'bracket', 'bounds', 'tol', 'options', 'maxiter')(self.defaults)

        if maxiter is not None:
            options = dict(options) if isinstance(options, Mapping) else {}
            options['maxiter'] = maxiter

        if self.defaults["prev_init"] and "x_prev" in self.global_state:
            if bracket is None: bracket = (0, 1)
            bracket = (*bracket[:-1], self.global_state["x_prev"])

        x = self.scopt.minimize_scalar(objective, method=method, bracket=bracket, bounds=bounds, tol=tol, options=options).x # pyright:ignore[reportAttributeAccessIssue]

        max = torch.finfo(var.params[0].dtype).max / 2
        if (not math.isfinite(x)) or abs(x) >= max: x = 0

        self.global_state['x_prev'] = x
        return x