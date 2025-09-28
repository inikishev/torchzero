from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
import pybobyqa

from ...utils import TensorList
from .wrapper import WrapperBase

Closure = Callable[[bool], Any]


class PyBobyqaWrapper(WrapperBase):
    """Use Py-BOBYQA is PyTorch optimizer.

    Note that this performs full minimization on each step,
    so usually you would want to perform a single step, although performing multiple steps will refine the
    solution.

    See https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/userguide.html for detailed descriptions of arguments.

    Args:
        params (Iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lb (float | None, optional): optional lower bounds. Defaults to None.
        ub (float | None, optional): optional upper bounds. Defaults to None.
        projections (list[Callable] | None, optional):
            a list of functions defining the Euclidean projections for each general convex constraint C_i.
            Each element of the list projections is a function that takes an input vector x (numpy array)
            and returns the closest point to that is in C_i. Defaults to None.
        npt (int | None, optional): the number of interpolation points to use. Defaults to None.
        rhobeg (float | None, optional):
            the initial value of the trust region radius. Defaults to None.
        rhoend (float | None, optional):
            minimum allowed value of trust region radius, which determines when a successful
            termination occurs. Defaults to 1e-8.
        maxfun (int | None, optional):
            the maximum number of objective evaluations the algorithm may request,
            default is min(100(n+1), 1000). Defaults to None.
        nsamples (Callable | None, optional):
            a Python function nsamples(delta, rho, iter, nrestarts)
            which returns the number of times to evaluate objfun at a given point.
            This is only applicable for objectives with stochastic noise,
            when averaging multiple evaluations at the same point produces a more accurate value.
            The input parameters are the trust region radius (delta),
            the lower bound on the trust region radius (rho),
            how many iterations the algorithm has been running for (iter),
            and how many restarts have been performed (nrestarts).
            Default is no averaging (i.e. nsamples(delta, rho, iter, nrestarts)=1).
            Defaults to None.
        user_params (dict | None, optional):
            dictionary of advanced parameters,
            see https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/advanced.html).
            Defaults to None.
        objfun_has_noise (bool, optional):
            a flag to indicate whether or not objfun has stochastic noise;
            i.e. will calling objfun(x) multiple times at the same value of x give different results?
            This is used to set some sensible default parameters (including using multiple restarts),
            all of which can be overridden by the values provided in user_params. Defaults to False.
        seek_global_minimum (bool, optional):
            a flag to indicate whether to search for a global minimum, rather than a local minimum.
            This is used to set some sensible default parameters,
            all of which can be overridden by the values provided in user_params.
            If True, both upper and lower bounds must be set.
            Note that Py-BOBYQA only implements a heuristic method,
            so there are no guarantees it will find a global minimum.
            However, by using this flag, it is more likely to escape local minima
            if there are better values nearby. The method used is a multiple restart mechanism,
            where we repeatedly re-initialize Py-BOBYQA from the best point found so far,
            but where we use a larger trust reigon radius each time
            (note: this is different to more common multi-start approach to global optimization).
            Defaults to False.
        scaling_within_bounds (bool, optional):
            a flag to indicate whether the algorithm should internally shift and scale the entries of x
            so that the bounds become 0 <= x <= 1. This is useful is you are setting bounds and the
            bounds have different orders of magnitude. If scaling_within_bounds=True,
            the values of rhobeg and rhoend apply to the shifted variables. Defaults to False.
        do_logging (bool, optional):
            a flag to indicate whether logging output should be produced.
            This is not automatically visible unless you use the Python logging module. Defaults to True.
        print_progress (bool, optional):
            a flag to indicate whether to print a per-iteration progress log to terminal. Defaults to False.
    """
    def __init__(
        self,
        params,
        lb: float | None = None,
        ub: float | None = None,
        projections = None,
        npt: int | None = None,
        rhobeg: float | None = None,
        rhoend: float = 1e-8,
        maxfun: int | None = None,
        nsamples: Callable | None | None = None,
        user_params: dict[str, Any] | None = None,
        objfun_has_noise: bool = False,
        seek_global_minimum: bool = False,
        scaling_within_bounds: bool = False,
        do_logging: bool = True,
        print_progress: bool = False,
    ):
        super().__init__(params, dict(lb=lb, ub=ub))
        kwargs = locals().copy()
        for k in ["self", "__class__", "params", "lb", "ub"]:
            del kwargs[k]
        self._kwargs = kwargs

    @torch.no_grad
    def step(self, closure: Closure):
        params = TensorList(self._get_params())
        x0 = params.to_vec().numpy(force=True)
        bounds = self._get_bounds()

        soln: pybobyqa.solver.OptimResults = pybobyqa.solve(
            objfun=partial(self._f, closure=closure, params=params),
            x0=x0,
            bounds=bounds,
            **self._kwargs
        )

        params.from_vec_(torch.as_tensor(soln.x, device = params[0].device, dtype=params[0].dtype,))
        return soln.f

