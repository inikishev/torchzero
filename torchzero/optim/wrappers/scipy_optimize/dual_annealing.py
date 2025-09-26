from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import scipy.optimize
import torch

from ....utils import TensorList
from ..wrapper import WrapperBase

Closure = Callable[[bool], Any]




class ScipyDualAnnealing(WrapperBase):
    def __init__(
        self,
        params,
        lb: float,
        ub: float,
        maxiter=1000,
        minimizer_kwargs=None,
        initial_temp=5230.0,
        restart_temp_ratio=2.0e-5,
        visit=2.62,
        accept=-5.0,
        maxfun=1e7,
        rng=None,
        no_local_search=False,
    ):
        super().__init__(params, dict(lb=lb, ub=ub))

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['lb'], kwargs['ub'], kwargs['__class__']
        self._kwargs = kwargs


    @torch.no_grad
    def step(self, closure: Closure):
        params = TensorList(self._get_params())
        x0 = params.to_vec().numpy(force=True)
        bounds = self._get_bounds()
        assert bounds is not None

        res = scipy.optimize.dual_annealing(
            partial(self._f, params = params, closure = closure),
            x0 = x0,
            bounds=bounds,
            **self._kwargs
        )

        params.from_vec_(torch.as_tensor(res.x, device = params[0].device, dtype=params[0].dtype))
        return res.fun
