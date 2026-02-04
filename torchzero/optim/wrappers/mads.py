from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
from mads.mads import orthomads

from ...utils import TensorList
from .wrapper import WrapperBase

Closure = Callable[[bool], Any]


class MADS(WrapperBase):
    """Use mads.orthomads as pytorch optimizer.

    Note that this performs full minimization on each step,
    so usually you would want to perform a single step, although performing multiple steps will refine the
    solution.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lb (float): lower bounds, this can also be specified in param_groups.
        ub (float): upper bounds, this can also be specified in param_groups.
        dp (float, optional): Initial poll size as percent of bounds. Defaults to 0.1.
        dm (float, optional): Initial mesh size as percent of bounds. Defaults to 0.01.
        dp_tol (float, optional): Minimum poll size stopping criteria. Defaults to -float('inf').
        nitermax (float, optional): Maximum objective function evaluations. Defaults to float('inf').
        displog (bool, optional): whether to show log. Defaults to False.
        savelog (bool, optional): whether to save log. Defaults to False.
    """
    def __init__(
        self,
        params,
        lb: float,
        ub: float,
        dp = 0.1,
        dm = 0.01,
        dp_tol = -float('inf'),
        nitermax = float('inf'),
        displog = False,
        savelog = False,
    ):
        super().__init__(params, dict(lb=lb, ub=ub))

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['lb'], kwargs['ub'], kwargs['__class__']
        self._kwargs = kwargs


    @torch.no_grad
    def step(self, closure: Closure):
        params = TensorList(self._get_params())
        x0 = params.to_vec().numpy(force=True)
        lb, ub = self._get_lb_ub()


        f, x = orthomads(
            design_variables=x0,
            bounds_upper=np.asarray(ub),
            bounds_lower=np.asarray(lb),
            objective_function=partial(self._f, params=params, closure=closure),
            **self._kwargs
        )

        params.from_vec_(torch.as_tensor(x, device = params[0].device, dtype=params[0].dtype,))
        return f

