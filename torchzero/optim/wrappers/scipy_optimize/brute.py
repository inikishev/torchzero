from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import scipy.optimize
import torch

from ....utils import TensorList
from ..wrapper import WrapperBase

Closure = Callable[[bool], Any]



class ScipyBrute(WrapperBase):
    def __init__(
        self,
        params,
        lb: float,
        ub: float,
        Ns: int = 20,
        finish = scipy.optimize.fmin,
        disp: bool = False,
        workers: int = 1
    ):
        super().__init__(params, dict(lb=lb,  ub=ub))

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['lb'], kwargs['ub'], kwargs['__class__']
        self._kwargs = kwargs

    @torch.no_grad
    def step(self, closure: Closure):
        params = TensorList(self._get_params())
        bounds = self._get_bounds()
        assert bounds is not None

        res,fval,grid,Jout = scipy.optimize.brute(
            partial(self._f, params = params, closure = closure),
            ranges=bounds,
            full_output=True,
            **self._kwargs
        )

        params.from_vec_(torch.as_tensor(res, device = params[0].device, dtype=params[0].dtype))

        return fval