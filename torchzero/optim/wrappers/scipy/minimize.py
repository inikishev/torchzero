from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import scipy.optimize
import torch

from ....utils import TensorList
from ..wrapper import WrapperBase

Closure = Callable[[bool], Any]


def _use_jac_hess_hessp(method, jac, hess, use_hessp):
    # those methods can't use hessp
    if (method is None) or (method.lower() not in ("newton-cg", "trust-ncg", "trust-krylov", "trust-constr")):
        use_hessp = False

    # those use gradients
    use_jac_autograd = (jac.lower() == 'autograd') and ((method is None) or (method.lower() in [
        'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'slsqp', 'dogleg',
        'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr',
    ]))

    # those use hessian/ some of them can use hessp instead
    use_hess_autograd = (isinstance(hess, str)) and (hess.lower() == 'autograd') and (method is not None) and (method.lower() in [
        'newton-cg', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact'
    ])

    # jac in scipy is '2-point', '3-point', 'cs', True or None.
    if jac == 'autograd':
        if use_jac_autograd: jac = True
        else: jac = None

    return jac, use_jac_autograd, use_hess_autograd, use_hessp

class ScipyMinimize(WrapperBase):
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
        use_hessp: bool = True,
    ):
        defaults = dict(lb=lb, ub=ub)
        super().__init__(params, defaults)
        self.method = method
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options
        self.hess = hess

        self.jac, self.use_jac_autograd, self.use_hess_autograd, self.use_hessp = _use_jac_hess_hessp(method, jac, hess, use_hessp)

    def _objective(self, x: np.ndarray, params: list[torch.Tensor], closure):
        if self.use_jac_autograd:
            f, g = self._f_g(x, params, closure)
            if self.method is not None and self.method.lower() == 'slsqp': g = g.astype(np.float64) #  slsqp requires float64
            return f, g

        return self._f(x, params, closure)

    def _hess(self, x: np.ndarray, params: list[torch.Tensor], closure):
        f,g,H = self._f_g_H(x, params, closure)
        return H

    def _hessp(self, x: np.ndarray, p:np.ndarray, params: list[torch.Tensor], closure):
        f,g,Hvp = self._f_g_Hvp(x, p, params, closure)
        return Hvp

    @torch.no_grad
    def step(self, closure: Closure):# pylint:disable = signature-differs # pyright:ignore[reportIncompatibleMethodOverride]
        params = TensorList(self._get_params())
        x0 = params.to_vec().numpy(force=True)
        bounds = self._get_bounds()

        # determine hess argument
        hess = self.hess
        hessp = None
        if hess == 'autograd':
            if self.use_hess_autograd:
                if self.use_hessp:
                    hessp = partial(self._hessp, params=params, closure=closure)
                    hess = None
                else:
                    hess = partial(self._hess, params=params, closure=closure)
            # hess = 'autograd' but method doesn't use hess
            else:
                hess = None


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
            hessp = hessp
        )

        params.from_vec_(torch.as_tensor(res.x, device = params[0].device, dtype=params[0].dtype))
        return res.fun
