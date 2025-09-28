from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import scipy.optimize
import torch

from ....utils import TensorList
from ..wrapper import WrapperBase

Closure = Callable[[bool], Any]




class ScipyDIRECT(WrapperBase):
    def __init__(
        self,
        params,
        lb: float,
        ub: float,
        maxfun: int | None = 1000,
        maxiter: int = 1000,
        eps: float = 0.0001,
        locally_biased: bool = True,
        f_min: float = -np.inf,
        f_min_rtol: float = 0.0001,
        vol_tol: float = 1e-16,
        len_tol: float = 0.000001,
        callback = None,
    ):
        super().__init__(params, dict(lb=lb, ub=ub))

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['lb'], kwargs['ub'], kwargs['__class__']
        self._kwargs = kwargs

    def _objective(self, x: np.ndarray, params: TensorList, closure) -> float:
        if self.raised: return np.inf
        try:
            return self._f(x, params, closure)

        except Exception as e:
            # this makes exceptions work in fcmaes and scipy direct
            self.e = e
            self.raised = True
            return np.inf

    @torch.no_grad
    def step(self, closure: Closure):
        self.raised = False
        self.e = None

        params = TensorList(self._get_params())
        bounds = self._get_bounds()
        assert bounds is not None

        res = scipy.optimize.direct(
            partial(self._objective, params=params, closure=closure),
            bounds=bounds,
            **self._kwargs
        )

        params.from_vec_(torch.as_tensor(res.x, device = params[0].device, dtype=params[0].dtype))

        if self.e is not None: raise self.e from None
        return res.fun

