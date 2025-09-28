from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import scipy.optimize
import torch

from ....utils import TensorList
from ..wrapper import WrapperBase
from .minimize import _use_jac_hess_hessp

Closure = Callable[[bool], Any]




class ScipyDualAnnealing(WrapperBase):
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
        method: Literal['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                    'l-bfgs-b', 'tnc', 'cobyla', 'cobyqa', 'slsqp',
                    'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact',
                    'trust-krylov'] | str = 'l-bfgs-b',
        jac: Literal['2-point', '3-point', 'cs', 'autograd'] = 'autograd',
        hess: Literal['2-point', '3-point', 'cs', 'autograd'] | scipy.optimize.HessianUpdateStrategy = 'autograd',
        use_hessp: bool = True,
    ):
        super().__init__(params, dict(lb=lb, ub=ub))

        kwargs = locals().copy()
        for k in ["self", "params", "lb", "ub", "__class__", "method", "jac", "hess", "use_hessp", "minimizer_kwargs"]:
            del kwargs[k]
        self._kwargs = kwargs

        self._minimizer_kwargs = minimizer_kwargs
        self.method = method
        self.hess = hess

        self.jac, self.use_jac_autograd, self.use_hess_autograd, self.use_hessp = _use_jac_hess_hessp(method, jac, hess, use_hessp)

    def _jac(self, x: np.ndarray, params: list[torch.Tensor], closure):
        f,g = self._f_g(x, params, closure)
        return g

    def _objective(self, x: np.ndarray, params: list[torch.Tensor], closure):
        # dual annealing doesn't support this
        # if self.use_jac_autograd:
        #     f, g = self._f_g(x, params, closure)
        #     if self.method.lower() == 'slsqp': g = g.astype(np.float64) #  slsqp requires float64
        #     return f, g

        return self._f(x, params, closure)

    def _hess(self, x: np.ndarray, params: list[torch.Tensor], closure):
        f,g,H = self._f_g_H(x, params, closure)
        return H

    def _hessp(self, x: np.ndarray, p:np.ndarray, params: list[torch.Tensor], closure):
        f,g,Hvp = self._f_g_Hvp(x, p, params, closure)
        return Hvp

    @torch.no_grad
    def step(self, closure: Closure):
        params = TensorList(self._get_params())
        x0 = params.to_vec().numpy(force=True)
        bounds = self._get_bounds()
        assert bounds is not None

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

        if self.method.lower() in ('tnc', 'slsqp'):
            x0 = x0.astype(np.float64) # those methods error without this

        minimizer_kwargs = self._minimizer_kwargs.copy() if self._minimizer_kwargs is not None else {}
        minimizer_kwargs.setdefault("method", self.method)
        minimizer_kwargs.setdefault("jac", partial(self._jac, params = params, closure = closure))
        minimizer_kwargs.setdefault("hess", hess)
        minimizer_kwargs.setdefault("hessp", hessp)
        minimizer_kwargs.setdefault("bounds", bounds)

        res = scipy.optimize.dual_annealing(
            partial(self._f, params = params, closure = closure),
            x0 = x0,
            bounds=bounds,
            minimizer_kwargs=minimizer_kwargs,
            **self._kwargs,
        )

        params.from_vec_(torch.as_tensor(res.x, device = params[0].device, dtype=params[0].dtype))
        return res.fun
