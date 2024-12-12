import typing as T
from collections import abc
from functools import partial

import numpy as np
import scipy.optimize
import torch

from ...core import ClosureType, TensorListOptimizer
from ...grad.derivatives import (hessian, hessian_list_to_mat,
                                 jacobian_and_hessian, jacobian_list_to_vec)
from ...modules import (SGD, Proj2Masks, ProjAscent, ProjGrad,
                        ProjGradAscentDifference, ProjLastAscentDifference,
                        ProjLastGradDifference, ProjNormalize, Subspace,
                        UninitializedClosureOptimizerWrapper)
from ...modules.subspace.random_subspace import Projection
from ...tensorlist import TensorList
from ..modular import ModularOptimizer


def _ensure_float(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().item()
    elif isinstance(x, np.ndarray): return x.item()
    return float(x)

class ScipyMinimize(TensorListOptimizer):
    def __init__(
        self,
        params,
        method=None,
        bounds=None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
        jac: T.Literal['2-point', '3-point', 'cs', 'autograd'] = 'autograd',
        hess: T.Literal['2-point', '3-point', 'cs', 'autograd'] | scipy.optimize.HessianUpdateStrategy = 'autograd'
    ):
        """Use scipy.minimize.optimize as pytorch optimizer. Note that this performs full minimization on each step,
        so usually you would want to perform a single step, although performing multiple steps will refine the
        solution.

        Please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        for a detailed description of args.

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups.
            method (_type_, optional): type of solver.
                If None, scipy will select one of BFGS, L-BFGS-B, SLSQP,
                depending on whether or not the problem has constraints or bounds.
                Defaults to None.
            bounds (_type_, optional): bounds on variables. Defaults to None.
            constraints (tuple, optional): constraints definition. Defaults to ().
            tol (_type_, optional): Tolerance for termination. Defaults to None.
            callback (_type_, optional): A callable called after each iteration. Defaults to None.
            options (_type_, optional): A dictionary of solver options. Defaults to None.
            jac (str, optional): Method for computing the gradient vector.
                Only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr.
                In addition to scipy options, this supports 'autograd', which uses pytorch autograd.
                This setting is ignored for methods that don't require gradient. Defaults to 'autograd'.
            hess (str, optional):
                Method for computing the Hessian matrix.
                Only for Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr.
                This setting is ignored for methods that don't require hessian. Defaults to 'autograd'.
        """
        super().__init__(params, {})
        self.method = method
        self.bounds = bounds
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options

        self.jac = jac
        self.hess = hess

        self.use_jac_autograd = jac.lower() == 'autograd' and (method is None or method.lower() in [
            'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'slsqp', 'dogleg',
            'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr',
        ])
        self.use_hess_autograd = isinstance(hess, str) and hess.lower() == 'autograd' and method is not None and method.lower() in [
            'newton-cg', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact'
        ]

        if self.jac == 'autograd':
            if self.use_jac_autograd: self.jac = True
            else: self.jac = None


    def _hess(self, x: np.ndarray, params: TensorList, closure: ClosureType): # type:ignore
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        with torch.enable_grad():
            value = closure(False)
            H = hessian([value], wrt = params) # type:ignore
        return hessian_list_to_mat(H).detach().cpu().numpy()

    def _objective(self, x: np.ndarray, params: TensorList, closure: ClosureType):
        # set params to x
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))

        # return value and maybe gradients
        if self.use_jac_autograd:
            with torch.enable_grad(): value = _ensure_float(closure(True))
            return value, params.grad.to_vec().detach().cpu().numpy()
        return _ensure_float(closure(False))

    @torch.no_grad
    def step(self, closure: ClosureType): # type:ignore # pylint:disable = signature-differs
        params = self.get_params()

        # determine hess argument
        if self.hess == 'autograd':
            if self.use_hess_autograd: hess = partial(self._hess, params = params, closure = closure)
            else: hess = None
        else: hess = self.hess

        x0 = params.to_vec().detach().cpu().numpy()

        if self.method is not None and (self.method.lower() == 'tnc' or self.method.lower() == 'slsqp'):
            x0 = x0.astype(np.float64) # those methods error without this

        res = scipy.optimize.minimize(
            partial(self._objective, params = params, closure = closure),
            x0 = x0,
            method=self.method,
            bounds=self.bounds,
            constraints=self.constraints,
            tol=self.tol,
            callback=self.callback,
            options=self.options,
            jac = self.jac,
            hess = hess,
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return res.fun


class ScipyDE(TensorListOptimizer):
    def __init__(
        self,
        params,
        bounds: tuple[float,float],
        strategy: str = 'best1bin',
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
        """Use scipy.minimize.differential_evolution as pytorch optimizer. Note that this performs full minimization on each step,
        so usually you would want to perform a single step. This also requires bounds to be specified.

        Please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
        for all other args.

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups.
            bounds (tuple[float,float], optional): tuple with lower and upper bounds.
                DE requires bounds to be specified. Defaults to None.

        other args: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
        """
        super().__init__(params, {})

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['bounds'], kwargs['__class__']
        self._kwargs = kwargs
        self._lb, self._ub = bounds

    def _objective(self, x: np.ndarray, params: TensorList, closure: ClosureType):
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return _ensure_float(closure(False))

    @torch.no_grad
    def step(self, closure: ClosureType): # type:ignore # pylint:disable = signature-differs
        params = self.get_params()

        x0 = params.to_vec().detach().cpu().numpy()
        bounds = [(self._lb, self._ub)] * len(x0)

        res = scipy.optimize.differential_evolution(
            partial(self._objective, params = params, closure = closure),
            x0 = x0,
            bounds=bounds,
            **self._kwargs
        )

        params.from_vec_(torch.from_numpy(res.x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return res.fun


class ScipyMinimizeSubspace(ModularOptimizer):
    def __init__(
        self,
        params,
        projections: Projection | abc.Iterable[Projection] = (
            Proj2Masks(5),
            ProjNormalize(
                ProjGrad(),
            )
        ),
        method=None,
        bounds=None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
        jac: T.Literal['2-point', '3-point', 'cs', 'autograd'] = 'autograd',
        hess: T.Literal['2-point', '3-point', 'cs', 'autograd'] | scipy.optimize.HessianUpdateStrategy = 'autograd',
    ):

        scopt = UninitializedClosureOptimizerWrapper(
                ScipyMinimize,
                method = method,
                bounds = bounds,
                constraints = constraints,
                tol = tol,
                callback = callback,
                options = options,
                jac = jac,
                hess = hess
            ),
        modules = [
            Subspace(scopt, projections),
        ]

        super().__init__(params, modules)