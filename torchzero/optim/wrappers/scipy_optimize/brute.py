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
        full_output: int = 0,
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

        res = scipy.optimize.brute(
            partial(self._f, params = params, closure = closure),
            ranges=bounds,
            **self._kwargs
        )

        x = res[0]
        fval = res[1]
        params.from_vec_(torch.as_tensor(x, device = params[0].device, dtype=params[0].dtype))

        return fval