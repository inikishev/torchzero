from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import scipy.optimize
import torch

from ....utils import TensorList
from ..wrapper import WrapperBase

Closure = Callable[[bool], Any]





class ScipyDE(WrapperBase):
    """Use scipy.minimize.differential_evolution as pytorch optimizer. Note that this performs full minimization on each step,
    so usually you would want to perform a single step. This also requires bounds to be specified.

    Please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    for all other args.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        bounds (tuple[float,float], optional): tuple with lower and upper bounds.
            DE requires bounds to be specified. Defaults to None.

        other args:
            refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    """
    def __init__(
        self,
        params,
        lb: float,
        ub: float,
        strategy: Literal['best1bin', 'best1exp', 'rand1bin', 'rand1exp', 'rand2bin', 'rand2exp',
            'randtobest1bin', 'randtobest1exp', 'currenttobest1bin', 'currenttobest1exp',
            'best2exp', 'best2bin'] = 'best1bin',
        maxiter: int = 1000,
        popsize: int = 15,
        tol: float = 0.01,
        mutation = (0.5, 1),
        recombination: float = 0.7,
        seed = None,
        callback = None,
        disp: bool = False,
        polish: bool = True,
        init: str = 'latinhypercube',
        atol: int = 0,
        updating: str = 'immediate',
        workers: int = 1,
        constraints = (),
        *,
        integrality = None,

    ):
        super().__init__(params, dict(lb=lb, ub=ub))

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['lb'], kwargs['ub'], kwargs['__class__']
        self._kwargs = kwargs

    @torch.no_grad
    def step(self, closure: Closure): # pylint:disable = signature-differs # pyright:ignore[reportIncompatibleMethodOverride]
        params = TensorList(self._get_params())
        x0 = params.to_vec().numpy(force=True)
        bounds = self._get_bounds()
        assert bounds is not None

        res = scipy.optimize.differential_evolution(
            partial(self._f, params = params, closure = closure),
            x0 = x0,
            bounds=bounds,
            **self._kwargs
        )

        params.from_vec_(torch.as_tensor(res.x, device = params[0].device, dtype=params[0].dtype))
        return res.fun
