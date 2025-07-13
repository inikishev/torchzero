from collections import abc
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import scipy.optimize
import torch

from ...utils import Optimizer, TensorList
from ...utils.derivatives import jacobian_and_hessian_mat_wrt, jacobian_wrt
from ...modules.second_order.newton import tikhonov_

def _ensure_float(x) -> float:
    if isinstance(x, torch.Tensor): return x.detach().cpu().item()
    if isinstance(x, np.ndarray): return float(x.item())
    return float(x)

def _ensure_numpy(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu()
    if isinstance(x, np.ndarray): return x
    return np.array(x)

def matrix_clamp(H: torch.Tensor, reg: float):
    try:
        eigvals, eigvecs = torch.linalg.eigh(H) # pylint:disable=not-callable
        eigvals.clamp_(min=reg)
        return eigvecs @ torch.diag(eigvals) @ eigvecs.mH
    except Exception:
        return H

Closure = Callable[[bool], Any]

class ScipyMinimize(Optimizer):
    """Use scipy.minimize.optimize as pytorch optimizer. Note that this performs full minimization on each step,
    so usually you would want to perform a single step, although performing multiple steps will refine the
    solution.

    Please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    for a detailed description of args.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        method (str | None, optional): type of solver.
            If None, scipy will select one of BFGS, L-BFGS-B, SLSQP,
            depending on whether or not the problem has constraints or bounds.
            Defaults to None.
        bounds (optional): bounds on variables. Defaults to None.
        constraints (tuple, optional): constraints definition. Defaults to ().
        tol (float | None, optional): Tolerance for termination. Defaults to None.
        callback (Callable | None, optional): A callable called after each iteration. Defaults to None.
        options (dict | None, optional): A dictionary of solver options. Defaults to None.
        jac (str, optional): Method for computing the gradient vector.
            Only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr.
            In addition to scipy options, this supports 'autograd', which uses pytorch autograd.
            This setting is ignored for methods that don't require gradient. Defaults to 'autograd'.
        hess (str, optional):
            Method for computing the Hessian matrix.
            Only for Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr.
            This setting is ignored for methods that don't require hessian. Defaults to 'autograd'.
        tikhonov (float, optional):
            optional hessian regularizer value. Only has effect for methods that require hessian.
    """
    def __init__(
        self,
        params,
        method: Literal['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                    'l-bfgs-b', 'tnc', 'cobyla', 'cobyqa', 'slsqp',
                    'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact',
                    'trust-krylov'] | str | None = None,
        lb = None,
        ub = None,
        constraints = (),
        tol: float | None = None,
        callback = None,
        options = None,
        jac: Literal['2-point', '3-point', 'cs', 'autograd'] = 'autograd',
        hess: Literal['2-point', '3-point', 'cs', 'autograd'] | scipy.optimize.HessianUpdateStrategy = 'autograd',
        tikhonov: float | None = 0,
        min_eigval: float | None = None,
    ):
        defaults = dict(lb=lb, ub=ub)
        super().__init__(params, defaults)
        self.method = method
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.min_eigval = min_eigval
        self.options = options

        self.jac = jac
        self.hess = hess
        self.tikhonov: float | None = tikhonov

        self.use_jac_autograd = jac.lower() == 'autograd' and (method is None or method.lower() in [
            'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'slsqp', 'dogleg',
            'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr',
        ])
        self.use_hess_autograd = isinstance(hess, str) and hess.lower() == 'autograd' and method is not None and method.lower() in [
            'newton-cg', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact'
        ]

        # jac in scipy is '2-point', '3-point', 'cs', True or None.
        if self.jac == 'autograd':
            if self.use_jac_autograd: self.jac = True
            else: self.jac = None


    def _hess(self, x: np.ndarray, params: TensorList, closure):
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        with torch.enable_grad():
            value = closure(False)
            _, H = jacobian_and_hessian_mat_wrt([value], wrt = params)
        if self.tikhonov is not None: H = tikhonov_(H, self.tikhonov)
        if self.min_eigval is not None: H = matrix_clamp(H, self.min_eigval)
        return H.detach().cpu().numpy()

    def _objective(self, x: np.ndarray, params: TensorList, closure):
        # set params to x
        params.from_vec_(torch.from_numpy(x).to(params[0], copy=False))

        # return value and maybe gradients
        if self.use_jac_autograd:
            with torch.enable_grad(): value = _ensure_float(closure())
            return value, params.ensure_grad_().grad.to_vec().detach().cpu().numpy()
        return _ensure_float(closure(False))

    @torch.no_grad
    def step(self, closure: Closure):# pylint:disable = signature-differs # pyright:ignore[reportIncompatibleMethodOverride]
        params = self.get_params()

        # determine hess argument
        if self.hess == 'autograd':
            if self.use_hess_autograd: hess = partial(self._hess, params = params, closure = closure)
            else: hess = None
        else: hess = self.hess

        x0 = params.to_vec().detach().cpu().numpy()

        # make bounds
        lb, ub = self.group_vals('lb', 'ub', cls=list)
        bounds = None
        if any(b is not None for b in lb) or any(b is not None for b in ub):
            bounds = []
            for p, l, u in zip(params, lb, ub):
                bounds.extend([(l, u)] * p.numel())

        if self.method is not None and (self.method.lower() == 'tnc' or self.method.lower() == 'slsqp'):
            x0 = x0.astype(np.float64) # those methods error without this

        res = scipy.optimize.minimize(
            partial(self._objective, params = params, closure = closure),
            x0 = x0,
            method=self.method,
            bounds=bounds,
            constraints=self.constraints,
            tol=self.tol,
            callback=self.callback,
            options=self.options,
            jac = self.jac,
            hess = hess,
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return res.fun



class ScipyRootOptimization(Optimizer):
    """Optimization via using scipy.root on gradients, mainly for experimenting!

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        method (str | None, optional): _description_. Defaults to None.
        tol (float | None, optional): _description_. Defaults to None.
        callback (_type_, optional): _description_. Defaults to None.
        options (_type_, optional): _description_. Defaults to None.
        jac (T.Literal[&#39;2, optional): _description_. Defaults to 'autograd'.
    """
    def __init__(
        self,
        params,
        method: Literal[
            "hybr",
            "lm",
            "broyden1",
            "broyden2",
            "anderson",
            "linearmixing",
            "diagbroyden",
            "excitingmixing",
            "krylov",
            "df-sane",
        ] = 'hybr',
        tol: float | None = None,
        callback = None,
        options = None,
        jac: Literal['2-point', '3-point', 'cs', 'autograd'] = 'autograd',
    ):
        super().__init__(params, {})
        self.method = method
        self.tol = tol
        self.callback = callback
        self.options = options

        self.jac = jac
        if self.jac == 'autograd': self.jac = True

        # those don't require jacobian
        if self.method.lower() in ('broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane'):
            self.jac = None

    def _objective(self, x: np.ndarray, params: TensorList, closure):
        # set params to x
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))

        # return gradients and maybe hessian
        if self.jac:
            with torch.enable_grad():
                self.value = closure(False)
                if not isinstance(self.value, torch.Tensor):
                    raise TypeError(f"Autograd jacobian requires closure to return torch.Tensor, got {type(self.value)}")
                g, H = jacobian_and_hessian_mat_wrt([self.value], wrt=params)
            return g.detach().cpu().numpy(), H.detach().cpu().numpy()

        # return the gradients
        with torch.enable_grad(): self.value = closure()
        jac = params.ensure_grad_().grad.to_vec()
        return jac.detach().cpu().numpy()

    @torch.no_grad
    def step(self, closure: Closure): # pylint:disable = signature-differs # pyright:ignore[reportIncompatibleMethodOverride]
        params = self.get_params()

        x0 = params.to_vec().detach().cpu().numpy()

        res = scipy.optimize.root(
            partial(self._objective, params = params, closure = closure),
            x0 = x0,
            method=self.method,
            tol=self.tol,
            callback=self.callback,
            options=self.options,
            jac = self.jac,
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return res.fun



class ScipyDE(Optimizer):
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
        polish: bool = False,
        init: str = 'latinhypercube',
        atol: int = 0,
        updating: str = 'immediate',
        workers: int = 1,
        constraints = (),
        *,
        integrality = None,

    ):
        super().__init__(params, lb=lb, ub=ub)

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['lb'], kwargs['ub'], kwargs['__class__']
        self._kwargs = kwargs

    def _objective(self, x: np.ndarray, params: TensorList, closure):
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return _ensure_float(closure(False))

    @torch.no_grad
    def step(self, closure: Closure): # pylint:disable = signature-differs # pyright:ignore[reportIncompatibleMethodOverride]
        params = self.get_params()

        x0 = params.to_vec().detach().cpu().numpy()

        lb, ub = self.group_vals('lb', 'ub', cls=list)
        bounds = []
        for p, l, u in zip(params, lb, ub):
            bounds.extend([(l, u)] * p.numel())

        res = scipy.optimize.differential_evolution(
            partial(self._objective, params = params, closure = closure),
            x0 = x0,
            bounds=bounds,
            **self._kwargs
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return res.fun



class ScipyDualAnnealing(Optimizer):
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
        bounds = []
        for p, l, u in zip(params, lb, ub):
            bounds.extend([(l, u)] * p.numel())

        res = scipy.optimize.dual_annealing(
            partial(self._objective, params = params, closure = closure),
            x0 = x0,
            bounds=bounds,
            **self._kwargs
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return res.fun



class ScipySHGO(Optimizer):
    def __init__(
        self,
        params,
        lb: float,
        ub: float,
        constraints = None,
        n: int = 100,
        iters: int = 1,
        callback = None,
        minimizer_kwargs = None,
        options = None,
        sampling_method: str = 'simplicial',
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

        lb, ub = self.group_vals('lb', 'ub', cls=list)
        bounds = []
        for p, l, u in zip(params, lb, ub):
            bounds.extend([(l, u)] * p.numel())

        res = scipy.optimize.shgo(
            partial(self._objective, params = params, closure = closure),
            bounds=bounds,
            **self._kwargs
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return res.fun


class ScipyDIRECT(Optimizer):
    def __init__(
        self,
        params,
        lb: float,
        ub: float,
        maxfun: int | None = 1000,
        maxiter: int = 1000,
        eps: float = 0.0001,
        locally_biased: bool = True,
        f_min: float = -np.inf,
        f_min_rtol: float = 0.0001,
        vol_tol: float = 1e-16,
        len_tol: float = 0.000001,
        callback = None,
    ):
        super().__init__(params, lb=lb, ub=ub)

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['lb'], kwargs['ub'], kwargs['__class__']
        self._kwargs = kwargs

    def _objective(self, x: np.ndarray, params: TensorList, closure) -> float:
        if self.raised: return np.inf
        try:
            params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
            return _ensure_float(closure(False))
        except Exception as e:
            # he he he ha, I found a way to make exceptions work in fcmaes and scipy direct
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
            bounds.extend([(l, u)] * p.numel())

        res = scipy.optimize.direct(
            partial(self._objective, params=params, closure=closure),
            bounds=bounds,
            **self._kwargs
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))

        if self.e is not None: raise self.e from None
        return res.fun




class ScipyBrute(Optimizer):
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

        lb, ub = self.group_vals('lb', 'ub', cls=list)
        bounds = []
        for p, l, u in zip(params, lb, ub):
            bounds.extend([(l, u)] * p.numel())

        x0 = scipy.optimize.brute(
            partial(self._objective, params = params, closure = closure),
            ranges=bounds,
            **self._kwargs
        )
        params.from_vec_(torch.from_numpy(x0).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return None
