from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import scipy.optimize
import torch

from ....utils import TensorList
from ..wrapper import WrapperBase

Closure = Callable[[bool], Any]


class ScipyRootOptimization(WrapperBase):

    """Optimization via using scipy.optimize.root on gradients, mainly for experimenting!

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

    def _objective(self, x: np.ndarray, params: list[torch.Tensor], closure):
        if self.jac:
            f, g, H = self._f_g_H(x, params, closure)
            return g, H

        f, g = self._f_g(x, params, closure)
        return g

    @torch.no_grad
    def step(self, closure: Closure): # pylint:disable = signature-differs # pyright:ignore[reportIncompatibleMethodOverride]
        params = TensorList(self._get_params())
        x0 = params.to_vec().numpy(force=True)

        res = scipy.optimize.root(
            partial(self._objective, params = params, closure = closure),
            x0 = x0,
            method=self.method,
            tol=self.tol,
            callback=self.callback,
            options=self.options,
            jac = self.jac,
        )

        params.from_vec_(torch.as_tensor(res.x, device = params[0].device, dtype=params[0].dtype))
        return res.fun


class ScipyLeastSquaresOptimization(WrapperBase):
    """Optimization via using scipy.optimize.least_squares on gradients, mainly for experimenting!

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
        method='trf',
        jac='autograd',
        bounds=(-np.inf, np.inf),
        ftol=1e-8, xtol=1e-8, gtol=1e-8, x_scale=1.0, loss='linear',
        f_scale=1.0, diff_step=None, tr_solver=None, tr_options=None,
        jac_sparsity=None, max_nfev=None, verbose=0
    ):
        super().__init__(params, {})
        kwargs = locals().copy()
        del kwargs['self'], kwargs['params'], kwargs['__class__'], kwargs['jac']
        self._kwargs = kwargs

        self.jac = jac


    def _objective(self, x: np.ndarray, params: list[torch.Tensor], closure):
        f, g = self._f_g(x, params, closure)
        return g

    def _hess(self, x: np.ndarray, params: list[torch.Tensor], closure):
        f,g,H = self._f_g_H(x, params, closure)
        return H

    @torch.no_grad
    def step(self, closure: Closure): # pylint:disable = signature-differs # pyright:ignore[reportIncompatibleMethodOverride]
        params = TensorList(self._get_params())
        x0 = params.to_vec().numpy(force=True)

        if self.jac == 'autograd': jac = partial(self._hess, params = params, closure = closure)
        else: jac = self.jac

        res = scipy.optimize.least_squares(
            partial(self._objective, params = params, closure = closure),
            x0 = x0,
            jac=jac, # type:ignore
            **self._kwargs
        )

        params.from_vec_(torch.as_tensor(res.x, device = params[0].device, dtype=params[0].dtype))
        return res.fun

