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


class ScipyBasinHopping(WrapperBase):
    def __init__(
        self,
        params,
        niter: int = 100,
        T: float = 1,
        stepsize: float = 0.5,
        minimizer_kwargs: dict | None = None,
        take_step: Callable | None = None,
        accept_test: Callable | None = None,
        callback: Callable | None = None,
        interval: int = 50,
        disp: bool = False,
        niter_success: int | None = None,
        rng: int | np.random.Generator | None = None,
        lb:float | None = None,
        ub:float | None = None,
        method: Literal['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                    'l-bfgs-b', 'tnc', 'cobyla', 'cobyqa', 'slsqp',
                    'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact',
                    'trust-krylov'] | str | None = None,
        jac: Literal['2-point', '3-point', 'cs', 'autograd'] = 'autograd',
        hess: Literal['2-point', '3-point', 'cs', 'autograd'] | scipy.optimize.HessianUpdateStrategy = 'autograd',
        use_hessp: bool = True,

        *,
        target_accept_rate: float = 0.5,
        stepwise_factor: float = 0.9
    ):
        super().__init__(params, dict(lb=lb, ub=ub))

        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['__class__'], kwargs["minimizer_kwargs"]
        del kwargs['method'], kwargs["jac"], kwargs['hess'], kwargs['use_hessp']
        del kwargs["lb"], kwargs["ub"]
        self._kwargs = kwargs

        self._minimizer_kwargs = minimizer_kwargs
        self.method = method
        self.hess = hess
        self.jac, self.use_jac_autograd, self.use_hess_autograd, self.use_hessp = _use_jac_hess_hessp(method, jac, hess, use_hessp)

    def _jac(self, x: np.ndarray, params: list[torch.Tensor], closure):
        f,g = self._f_g(x, params, closure)
        return g

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
    def step(self, closure: Closure):
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

        minimizer_kwargs = self._minimizer_kwargs.copy() if self._minimizer_kwargs is not None else {}
        minimizer_kwargs.setdefault("method", self.method)
        minimizer_kwargs.setdefault("jac", self.jac)
        minimizer_kwargs.setdefault("hess", hess)
        minimizer_kwargs.setdefault("hessp", hessp)
        minimizer_kwargs.setdefault("bounds", bounds)

        res = scipy.optimize.basinhopping(
            partial(self._objective, params = params, closure = closure),
            x0 = params.to_vec().numpy(force=True),
            minimizer_kwargs=minimizer_kwargs,
            **self._kwargs
        )

        params.from_vec_(torch.as_tensor(res.x, device = params[0].device, dtype=params[0].dtype))
        return res.fun
