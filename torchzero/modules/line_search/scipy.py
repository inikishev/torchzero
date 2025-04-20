from collections.abc import Mapping
from operator import itemgetter

import torch

from .line_search import LineSearch


class ScipyMinimizeScalar(LineSearch):
    def __init__(
        self,
        method: str | None = None,
        maxiter: int | None = None,
        bracket=None,
        bounds=None,
        tol: float | None = None,
        options=None,
    ):
        defaults = dict(method=method,bracket=bracket,bounds=bounds,tol=tol,options=options,maxiter=maxiter)
        super().__init__(defaults)

        import scipy.optimize
        self.scopt = scipy.optimize


    @torch.no_grad
    def search(self, update, vars):
        objective = self.make_objective(vars=vars)
        method, bracket, bounds, tol, options, maxiter = itemgetter(
            'method', 'bracket', 'bounds', 'tol', 'options', 'maxiter')(self.defaults)

        if maxiter is not None:
            options = dict(options) if isinstance(options, Mapping) else {}
            options['maxiter'] = maxiter

        res = self.scopt.minimize_scalar(objective, method=method, bracket=bracket, bounds=bounds, tol=tol, options=options)
        return res.x