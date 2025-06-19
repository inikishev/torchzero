from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
from mads.mads import orthomads

from ...utils import Optimizer, TensorList


def _ensure_float(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().item()
    if isinstance(x, np.ndarray): return x.item()
    return float(x)

def _ensure_numpy(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu()
    if isinstance(x, np.ndarray): return x
    return np.array(x)


Closure = Callable[[bool], Any]


class MADS(Optimizer):
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
        super().__init__(params, lb=lb, ub=ub)

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['lb'], kwargs['ub'], kwargs['__class__']
        self._kwargs = kwargs

    def _objective(self, x: np.ndarray, params: TensorList, closure):
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return _ensure_float(closure(False))

    @torch.no_grad
    def step(self, closure: Closure):
        params = self.get_params()

        x0 = params.to_vec().detach().cpu().numpy()

        lb, ub = self.group_vals('lb', 'ub', cls=list)
        bounds_lower = []
        bounds_upper = []
        for p, l, u in zip(params, lb, ub):
            bounds_lower.extend([l] * p.numel())
            bounds_upper.extend([u] * p.numel())

        f, x = orthomads(
            design_variables=x0,
            bounds_upper=np.asarray(bounds_upper),
            bounds_lower=np.asarray(bounds_lower),
            objective_function=partial(self._objective, params = params, closure = closure),
            **self._kwargs
        )

        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return f

