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


class ScipySHGO(WrapperBase):
    def __init__(
        self,
        params,
        lb: float,
        ub: float,
        constraints = None,
        n: int = 100,
        iters: int = 1,
        callback = None,
        options: dict | None = None,
        sampling_method: str = 'simplicial',
        minimizer_kwargs: dict | None = None,
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
        del kwargs['self'], kwargs['params'], kwargs['lb'], kwargs['ub'], kwargs['__class__'], kwargs["options"]
        del kwargs["method"], kwargs["jac"], kwargs["hess"], kwargs["use_hessp"], kwargs["minimizer_kwargs"]
        self._kwargs = kwargs
        self.minimizer_kwargs = minimizer_kwargs
        self.options = options
        self.method = method
        self.hess = hess

        self.jac, self.use_jac_autograd, self.use_hess_autograd, self.use_hessp = _use_jac_hess_hessp(method, jac, hess, use_hessp)


    def _objective(self, x: np.ndarray, params: list[torch.Tensor], closure):
        if self.use_jac_autograd:
            f, g = self._f_g(x, params, closure)
            if self.method.lower() == 'slsqp': g = g.astype(np.float64) #  slsqp requires float64
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

        minimizer_kwargs = self.minimizer_kwargs.copy() if self.minimizer_kwargs is not None else {}
        minimizer_kwargs.setdefault("method", self.method)

        options = self.options.copy() if self.options is not None else {}
        minimizer_kwargs.setdefault("jac", self.jac)
        minimizer_kwargs.setdefault("hess", hess)
        minimizer_kwargs.setdefault("hessp", hessp)
        minimizer_kwargs.setdefault("bounds", bounds)

        res = scipy.optimize.shgo(
            partial(self._objective, params=params, closure=closure),
            bounds=bounds,
            minimizer_kwargs=minimizer_kwargs,
            options=options,
            **self._kwargs
        )

        params.from_vec_(torch.as_tensor(res.x, device = params[0].device, dtype=params[0].dtype))
        return res.fun

