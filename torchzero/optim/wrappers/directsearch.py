from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import directsearch
import numpy as np
import torch
from directsearch.ds import DEFAULT_PARAMS

from ...modules.second_order.newton import tikhonov_
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


class DirectSearch(Optimizer):
    """Use directsearch as pytorch optimizer.

    Note that this performs full minimization on each step,
    so usually you would want to perform a single step, although performing multiple steps will refine the
    solution.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.

        rho: Choice of the forcing function.

        sketch_dim: Reduced dimension to generate polling directions in.

        sketch_type: Sketching technique to be used.

        maxevals: Maximum number of calls to f performed by the algorithm.

        poll_type: Type of polling directions generated in the reduced spaces.

        alpha0: Initial value for the stepsize parameter.

        alpha_max: Maximum value for the stepsize parameter.

        alpha_min: Minimum value for the stepsize parameter.

        gamma_inc: Increase factor for the stepsize update.

        gamma_dec: Decrease factor for the stepsize update.

        verbose:
            Boolean indicating whether information should be displayed during an algorithmic run.

        print_freq:
            Value indicating how frequently information should be displayed.

        use_stochastic_three_points:
            Boolean indicating whether the specific stochastic three points method should be used.

        poll_scale_prob: Probability of scaling the polling directions.

        poll_scale_factor: Factor used to scale the polling directions.

        rho_uses_normd:
            Boolean indicating whether the forcing function should account for the norm of the direction.


    """
    def __init__(
        self,
        params,
        maxevals = DEFAULT_PARAMS['maxevals'], # Maximum number of function evaluations
        rho = DEFAULT_PARAMS['rho'], # Forcing function
        sketch_dim = DEFAULT_PARAMS['sketch_dim'], # Target dimension for sketching
        sketch_type = DEFAULT_PARAMS['sketch_type'], # Sketching technique
        poll_type = DEFAULT_PARAMS['poll_type'], # Polling direction type
        alpha0 = DEFAULT_PARAMS['alpha0'], # Original stepsize value
        alpha_max = DEFAULT_PARAMS['alpha_max'], # Maximum value for the stepsize
        alpha_min = DEFAULT_PARAMS['alpha_min'], # Minimum value for the stepsize
        gamma_inc = DEFAULT_PARAMS['gamma_inc'], # Increasing factor for the stepsize
        gamma_dec = DEFAULT_PARAMS['gamma_dec'], # Decreasing factor for the stepsize
        verbose = DEFAULT_PARAMS['verbose'], # Display information about the method
        print_freq = DEFAULT_PARAMS['print_freq'], # How frequently to display information
        use_stochastic_three_points = DEFAULT_PARAMS['use_stochastic_three_points'], # Boolean for a specific method
        rho_uses_normd = DEFAULT_PARAMS['rho_uses_normd'], # Forcing function based on direction norm
    ):
        super().__init__(params, {})

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['__class__']
        self._kwargs = kwargs

    def _objective(self, x: np.ndarray, params: TensorList, closure):
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return _ensure_float(closure(False))

    @torch.no_grad
    def step(self, closure: Closure):
        params = self.get_params()

        x0 = params.to_vec().detach().cpu().numpy()

        res = directsearch.solve(
            partial(self._objective, params = params, closure = closure),
            x0 = x0,
            **self._kwargs
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return res.f



class DirectSearchDS(Optimizer):
    def __init__(
        self,
        params,
        maxevals = DEFAULT_PARAMS['maxevals'], # Maximum number of function evaluations
        rho = DEFAULT_PARAMS['rho'], # Forcing function
        poll_type = DEFAULT_PARAMS['poll_type'], # Polling direction type
        alpha0 = DEFAULT_PARAMS['alpha0'], # Original stepsize value
        alpha_max = DEFAULT_PARAMS['alpha_max'], # Maximum value for the stepsize
        alpha_min = DEFAULT_PARAMS['alpha_min'], # Minimum value for the stepsize
        gamma_inc = DEFAULT_PARAMS['gamma_inc'], # Increasing factor for the stepsize
        gamma_dec = DEFAULT_PARAMS['gamma_dec'], # Decreasing factor for the stepsize
        verbose = DEFAULT_PARAMS['verbose'], # Display information about the method
        print_freq = DEFAULT_PARAMS['print_freq'], # How frequently to display information
        rho_uses_normd = DEFAULT_PARAMS['rho_uses_normd'], # Forcing function based on direction norm
    ):
        super().__init__(params, {})

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['__class__']
        self._kwargs = kwargs

    def _objective(self, x: np.ndarray, params: TensorList, closure):
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return _ensure_float(closure(False))

    @torch.no_grad
    def step(self, closure: Closure):
        params = self.get_params()

        x0 = params.to_vec().detach().cpu().numpy()

        res = directsearch.solve_directsearch(
            partial(self._objective, params = params, closure = closure),
            x0 = x0,
            **self._kwargs
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return res.f

class DirectSearchProbabilistic(Optimizer):
    def __init__(
        self,
        params,
        maxevals = DEFAULT_PARAMS['maxevals'], # Maximum number of function evaluations
        rho = DEFAULT_PARAMS['rho'], # Forcing function
        alpha0 = DEFAULT_PARAMS['alpha0'], # Original stepsize value
        alpha_max = DEFAULT_PARAMS['alpha_max'], # Maximum value for the stepsize
        alpha_min = DEFAULT_PARAMS['alpha_min'], # Minimum value for the stepsize
        gamma_inc = DEFAULT_PARAMS['gamma_inc'], # Increasing factor for the stepsize
        gamma_dec = DEFAULT_PARAMS['gamma_dec'], # Decreasing factor for the stepsize
        verbose = DEFAULT_PARAMS['verbose'], # Display information about the method
        print_freq = DEFAULT_PARAMS['print_freq'], # How frequently to display information
        rho_uses_normd = DEFAULT_PARAMS['rho_uses_normd'], # Forcing function based on direction norm
    ):
        super().__init__(params, {})

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['__class__']
        self._kwargs = kwargs

    def _objective(self, x: np.ndarray, params: TensorList, closure):
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return _ensure_float(closure(False))

    @torch.no_grad
    def step(self, closure: Closure):
        params = self.get_params()

        x0 = params.to_vec().detach().cpu().numpy()

        res = directsearch.solve_probabilistic_directsearch(
            partial(self._objective, params = params, closure = closure),
            x0 = x0,
            **self._kwargs
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return res.f


class DirectSearchSubspace(Optimizer):
    def __init__(
        self,
        params,
        maxevals = DEFAULT_PARAMS['maxevals'], # Maximum number of function evaluations
        rho = DEFAULT_PARAMS['rho'], # Forcing function
        sketch_dim = DEFAULT_PARAMS['sketch_dim'], # Target dimension for sketching
        sketch_type = DEFAULT_PARAMS['sketch_type'], # Sketching technique
        poll_type = DEFAULT_PARAMS['poll_type'], # Polling direction type
        alpha0 = DEFAULT_PARAMS['alpha0'], # Original stepsize value
        alpha_max = DEFAULT_PARAMS['alpha_max'], # Maximum value for the stepsize
        alpha_min = DEFAULT_PARAMS['alpha_min'], # Minimum value for the stepsize
        gamma_inc = DEFAULT_PARAMS['gamma_inc'], # Increasing factor for the stepsize
        gamma_dec = DEFAULT_PARAMS['gamma_dec'], # Decreasing factor for the stepsize
        verbose = DEFAULT_PARAMS['verbose'], # Display information about the method
        print_freq = DEFAULT_PARAMS['print_freq'], # How frequently to display information
        rho_uses_normd = DEFAULT_PARAMS['rho_uses_normd'], # Forcing function based on direction norm
    ):
        super().__init__(params, {})

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['__class__']
        self._kwargs = kwargs

    def _objective(self, x: np.ndarray, params: TensorList, closure):
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return _ensure_float(closure(False))

    @torch.no_grad
    def step(self, closure: Closure):
        params = self.get_params()

        x0 = params.to_vec().detach().cpu().numpy()

        res = directsearch.solve_subspace_directsearch(
            partial(self._objective, params = params, closure = closure),
            x0 = x0,
            **self._kwargs
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return res.f



class DirectSearchSTP(Optimizer):
    def __init__(
        self,
        params,
        maxevals = DEFAULT_PARAMS['maxevals'], # Maximum number of function evaluations
        alpha0 = DEFAULT_PARAMS['alpha0'], # Original stepsize value
        alpha_min = DEFAULT_PARAMS['alpha_min'], # Minimum value for the stepsize
        verbose = DEFAULT_PARAMS['verbose'], # Display information about the method
        print_freq = DEFAULT_PARAMS['print_freq'], # How frequently to display information
    ):
        super().__init__(params, {})

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['__class__']
        self._kwargs = kwargs

    def _objective(self, x: np.ndarray, params: TensorList, closure):
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return _ensure_float(closure(False))

    @torch.no_grad
    def step(self, closure: Closure):
        params = self.get_params()

        x0 = params.to_vec().detach().cpu().numpy()

        res = directsearch.solve_stp(
            partial(self._objective, params = params, closure = closure),
            x0 = x0,
            **self._kwargs
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return res.f