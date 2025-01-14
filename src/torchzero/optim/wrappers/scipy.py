from typing import Literal, Any
from collections import abc
from functools import partial

import numpy as np
import torch

import scipy.optimize

from ...core import _ClosureType, TensorListOptimizer
from ...utils.derivatives import jacobian, jacobian_list_to_vec, hessian, hessian_list_to_mat, jacobian_and_hessian
from ...modules import Wrap
from ...modules.experimental.subspace import Projection, Proj2Masks, ProjGrad, ProjNormalize, Subspace
from ...modules.second_order.newton import regularize_hessian_
from ...tensorlist import TensorList
from ..modular import Modular


def _ensure_float(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().item()
    if isinstance(x, np.ndarray): return x.item()
    return float(x)

def _ensure_numpy(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu()
    if isinstance(x, np.ndarray): return x
    return np.array(x)

class ScipyMinimize(TensorListOptimizer):
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
        tikhonov: float | Literal['eig'] = 0,
    ):
        defaults = dict(lb=lb, ub=ub)
        super().__init__(params, defaults)
        self.method = method
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options

        self.jac = jac
        self.hess = hess
        self.tikhonov: float | Literal['eig'] = tikhonov

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


    def _hess(self, x: np.ndarray, params: TensorList, closure: _ClosureType): # type:ignore
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        with torch.enable_grad():
            value = closure(False)
            H = hessian([value], wrt = params) # type:ignore
        Hmat =  hessian_list_to_mat(H)
        regularize_hessian_(Hmat, self.tikhonov)
        return Hmat.detach().cpu().numpy()

    def _objective(self, x: np.ndarray, params: TensorList, closure: _ClosureType):
        # set params to x
        params.from_vec_(torch.from_numpy(x).to(params[0], copy=False))

        # return value and maybe gradients
        if self.use_jac_autograd:
            with torch.enable_grad(): value = _ensure_float(closure())
            return value, params.ensure_grad_().grad.to_vec().detach().cpu().numpy()
        return _ensure_float(closure(False))

    @torch.no_grad
    def step(self, closure: _ClosureType): # type:ignore # pylint:disable = signature-differs
        params = self.get_params()

        # determine hess argument
        if self.hess == 'autograd':
            if self.use_hess_autograd: hess = partial(self._hess, params = params, closure = closure)
            else: hess = None
        else: hess = self.hess

        x0 = params.to_vec().detach().cpu().numpy()

        # make bounds
        lb, ub = self.get_group_keys(['lb', 'ub'], cls=list)
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



class ScipyRoot(TensorListOptimizer):
    """Find a root of a vector function (UNTESTED!).

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

    def _objective(self, x: np.ndarray, params: TensorList, closure: _ClosureType):
        # set params to x
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))

        # return value and maybe gradients
        if self.jac:
            with torch.enable_grad():
                value = closure(False)
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Autograd jacobian requires closure to return torch.Tensor, got {type(value)}")
            jac = jacobian_list_to_vec(jacobian([value], wrt=params))
            return _ensure_numpy(value), jac.detach().cpu().numpy()
        return _ensure_numpy(closure(False))

    @torch.no_grad
    def step(self, closure: _ClosureType): # type:ignore # pylint:disable = signature-differs
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


class ScipyRootOptimization(TensorListOptimizer):
    """Optimization via finding roots of the gradient with `scipy.optimize.root` (for experiments, won't work well on most problems).

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        method (str, optional): one of methods from https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html#scipy.optimize.root. Defaults to 'hybr'.
        tol (float | None, optional): tolerance. Defaults to None.
        callback (_type_, optional): callback. Defaults to None.
        options (_type_, optional): options for optimizer. Defaults to None.
        jac (Literal[&#39;2, optional): jacobian calculation method. Defaults to 'autograd'.
        tikhonov (float | Literal[&#39;eig&#39;], optional): tikhonov regularization (only for 'hybr' and 'lm'). Defaults to 0.
        add_loss (float, optional): adds loss value to jacobian multiplied by this to try to avoid finding maxima. Defaults to 0.
        mul_loss (float, optional): multiplies jacobian by loss value multiplied by this to try to avoid finding maxima. Defaults to 0.
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
        tikhonov: float | Literal['eig'] = 0,
        add_loss: float = 0,
        mul_loss: float = 0,
    ):
        super().__init__(params, {})
        self.method = method
        self.tol = tol
        self.callback = callback
        self.options = options
        self.value = None
        self.tikhonov: float | Literal['eig'] = tikhonov
        self.add_loss = add_loss
        self.mul_loss = mul_loss

        self.jac = jac == 'autograd'

        # those don't require jacobian
        if self.method.lower() in ('broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane'):
            self.jac = None

    def _objective(self, x: np.ndarray, params: TensorList, closure: _ClosureType):
        # set params to x
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))

        # return gradients and maybe hessian
        if self.jac:
            with torch.enable_grad():
                self.value = closure(False)
                if not isinstance(self.value, torch.Tensor):
                    raise TypeError(f"Autograd jacobian requires closure to return torch.Tensor, got {type(self.value)}")
                jac_list, hess_list = jacobian_and_hessian([self.value], wrt=params)
            jac = jacobian_list_to_vec(jac_list)
            hess = hessian_list_to_mat(hess_list)
            regularize_hessian_(hess, self.tikhonov)
            if self.mul_loss != 0: jac *= self.value * self.mul_loss
            if self.add_loss != 0: jac += self.value * self.add_loss
            return jac.detach().cpu().numpy(), hess.detach().cpu().numpy()

        # return the gradients
        with torch.enable_grad(): self.value = closure()
        jac = params.ensure_grad_().grad.to_vec()
        if self.mul_loss != 0: jac *= self.value * self.mul_loss
        if self.add_loss != 0: jac += self.value * self.add_loss
        return jac.detach().cpu().numpy()

    @torch.no_grad
    def step(self, closure: _ClosureType): # type:ignore # pylint:disable = signature-differs
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
        return self.value

class ScipyDE(TensorListOptimizer):
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
        bounds: tuple[float,float],
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
        super().__init__(params, {})

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['bounds'], kwargs['__class__']
        self._kwargs = kwargs
        self._lb, self._ub = bounds

    def _objective(self, x: np.ndarray, params: TensorList, closure: _ClosureType):
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))
        return _ensure_float(closure(False))

    @torch.no_grad
    def step(self, closure: _ClosureType): # type:ignore # pylint:disable = signature-differs
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


class ScipyMinimizeSubspace(Modular):
    """for experiments and won't work well on most problems.

    explanation - optimizes in a small subspace using scipy.optimize.minimize, but doesnt seem to work well"""
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
        lb = None,
        ub = None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
        jac: Literal['2-point', '3-point', 'cs', 'autograd'] = 'autograd',
        hess: Literal['2-point', '3-point', 'cs', 'autograd'] | scipy.optimize.HessianUpdateStrategy = '2-point',
    ):

        scopt = Wrap(
                ScipyMinimize,
                pass_closure = True,
                method = method,
                lb = lb,
                ub = ub,
                constraints = constraints,
                tol = tol,
                callback = callback,
                options = options,
                jac = jac,
                hess = hess
            )
        modules = [
            Subspace(scopt, projections),
        ]

        super().__init__(params, modules)