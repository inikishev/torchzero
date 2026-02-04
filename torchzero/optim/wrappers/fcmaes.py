from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import torch

import fcmaes
import fcmaes.optimizer
import fcmaes.retry

from ...utils import TensorList
from .wrapper import WrapperBase

Closure = Callable[[bool], Any]

def silence_fcmaes():
    fcmaes.retry.logger.disable('fcmaes')

class FcmaesWrapper(WrapperBase):
    """Use fcmaes as pytorch optimizer. Particularly fcmaes has BITEOPT which appears to win in many benchmarks.

    Note that this performs full minimization on each step, so only perform one step with this.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lb (float): lower bounds, this can also be specified in param_groups.
        ub (float): upper bounds, this can also be specified in param_groups.
        optimizer (fcmaes.optimizer.Optimizer | None, optional):
            optimizer to use. Default is a sequence of differential evolution and CMA-ES.
        max_evaluations (int | None, optional):
            Forced termination of all optimization runs after `max_evaluations` function evaluations.
            Only used if optimizer is undefined, otherwise this setting is defined in the optimizer. Defaults to 50000.
        value_limit (float | None, optional): Upper limit for optimized function values to be stored. Defaults to np.inf.
        num_retries (int | None, optional): Number of optimization retries. Defaults to 1.
        popsize (int | None, optional):
            CMA-ES population size used for all CMA-ES runs.
            Not used for differential evolution.
            Ignored if parameter optimizer is defined. Defaults to 31.
        capacity (int | None, optional): capacity of the evaluation store. Defaults to 500.
        stop_fitness (float | None, optional):
            Limit for fitness value. optimization runs terminate if this value is reached. Defaults to -np.inf.
        statistic_num (int | None, optional):
            if > 0 stores the progress of the optimization. Defines the size of this store. Defaults to 0.
    """
    def __init__(
        self,
        params,
        lb: float,
        ub: float,
        optimizer: fcmaes.optimizer.Optimizer | None = None,
        max_evaluations: int | None = 50000,
        value_limit: float | None = np.inf,
        num_retries: int | None = 1,
        # workers: int = 1,
        popsize: int | None = 31,
        capacity: int | None = 500,
        stop_fitness: float | None = -np.inf,
        statistic_num: int | None = 0,
        silence: bool = True,
    ):
        super().__init__(params, dict(lb=lb,ub=ub))
        if silence:
            silence_fcmaes()
        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['lb'], kwargs['ub'], kwargs['__class__'], kwargs["silence"]
        self._kwargs = kwargs
        self._kwargs['workers'] = 1


    @torch.no_grad
    def step(self, closure: Closure):

        params = TensorList(self._get_params())
        bounds = self._get_bounds()

        res = fcmaes.retry.minimize(
            partial(self._f, params=params, closure=closure), # pyright:ignore[reportArgumentType]
            bounds=bounds, # pyright:ignore[reportArgumentType]
            **self._kwargs
        )

        params.from_vec_(torch.as_tensor(res.x, device = params[0].device, dtype=params[0].dtype))
        return res.fun

