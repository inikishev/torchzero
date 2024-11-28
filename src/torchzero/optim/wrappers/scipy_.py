import typing as T
from collections import abc
from functools import partial

import numpy as np
import scipy.optimize
import torch

from ...core import ClosureType, TensorListOptimizer
from ...grad.derivatives import (hessian, hessian_list_to_mat,
                                 jacobian_and_hessian, jacobian_list_to_vec)
from ...tensorlist import TensorList


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

        res = scipy.optimize.minimize(
            partial(self._objective, params = params, closure = closure),
            params.to_vec().detach().cpu().numpy(),
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

