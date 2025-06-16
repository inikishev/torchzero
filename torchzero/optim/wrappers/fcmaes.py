from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import fcmaes
import fcmaes.optimizer
import fcmaes.retry
import numpy as np
import torch

from ...utils import Optimizer, TensorList

Closure = Callable[[bool], Any]


def _ensure_float(x) -> float:
    if isinstance(x, torch.Tensor): return x.detach().cpu().item()
    if isinstance(x, np.ndarray): return float(x.item())
    return float(x)

def silence_fcmaes():
    fcmaes.retry.logger.disable('fcmaes')

class FcmaesWrapper(Optimizer):
    """Use fcmaes as pytorch optimizer. Particularly fcmaes has BITEOPT which appears to win in many benchmarks.

    Note that this performs full minimization on each step, so only perform one step with this.

    Args:
        params (_type_): _description_
        lb (float): _description_
        ub (float): _description_
        optimizer (fcmaes.optimizer.Optimizer | None, optional): _description_. Defaults to None.
        max_evaluations (int | None, optional): _description_. Defaults to 50000.
        value_limit (float | None, optional): _description_. Defaults to np.inf.
        num_retries (int | None, optional): _description_. Defaults to 1.
        workers (int, optional): _description_. Defaults to 1.
        popsize (int | None, optional): _description_. Defaults to 31.
        capacity (int | None, optional): _description_. Defaults to 500.
        stop_fitness (float | None, optional): _description_. Defaults to -np.inf.
        statistic_num (int | None, optional): _description_. Defaults to 0.
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
        workers: int = 1,
        popsize: int | None = 31,
        capacity: int | None = 500,
        stop_fitness: float | None = -np.inf,
        statistic_num: int | None = 0
    ):
        super().__init__(params, lb=lb, ub=ub)
        silence_fcmaes()
        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['lb'], kwargs['ub'], kwargs['__class__']
        self._kwargs = kwargs

    def _objective(self, x: np.ndarray, params: TensorList, closure) -> float:
        if self.raised: return np.inf
        try:
            params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
            return _ensure_float(closure(False))
        except Exception as e:
            # ha ha, I found a way to make exceptions work in fcmaes and scipy direct
            self.e = e
            self.raised = True
            return np.inf

    @torch.no_grad
    def step(self, closure: Closure):
        self.raised = False
        self.e = None

        params = self.get_params()

        lb, ub = self.group_vals('lb', 'ub', cls=list)
        bounds = []
        for p, l, u in zip(params, lb, ub):
            bounds.extend([[l, u]] * p.numel())

        res = fcmaes.retry.minimize(
            partial(self._objective, params=params, closure=closure), # pyright:ignore[reportArgumentType]
            bounds=bounds, # pyright:ignore[reportArgumentType]
            **self._kwargs
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))

        if self.e is not None: raise self.e from None
        return res.fun

