import itertools
import math
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from typing import Any, Literal

import numpy as np
import scipy.optimize
import torch

from ...core import Chainable, Module, apply_transform
from ...utils import TensorList, vec_to_tensors, vec_to_tensors_
from ...utils.derivatives import (
    hessian_list_to_mat,
    jacobian_wrt,
    hvp,
)

def _poly_eval_diag(s: np.ndarray, c, derivatives):
    val = float(c) + (derivatives[0] * s).sum(-1)

    if len(derivatives) > 1:
        for i, d_diag in enumerate(derivatives[1:], 2):
            val += (d_diag * (s**i)).sum(-1) / math.factorial(i)

    return val

def _proximal_poly_v_diag(x: np.ndarray, c, prox, x0: np.ndarray, derivatives):
    """Computes the value of the proximal polynomial approximation."""
    if x.ndim == 2: x = x.T
    s = x - x0

    val = _poly_eval_diag(s, c, derivatives)

    penalty = 0
    if prox != 0:
        penalty = (prox / 2) * (s**2).sum(-1)

    return val + penalty

def _proximal_poly_g_diag(x: np.ndarray, c, prox, x0: np.ndarray, derivatives):
    """Computes the gradient of the proximal polynomial approximation."""
    s = x - x0

    g = derivatives[0].copy()

    if len(derivatives) > 1:
        for i, d_diag in enumerate(derivatives[1:], 2):
            g += d_diag * (s**(i - 1)) / math.factorial(i - 1)

    if prox != 0:
        g += prox * s

    return g

def _proximal_poly_H_diag(x: np.ndarray, c, prox, x0: np.ndarray, derivatives):
    """Computes the Hessian of the proximal polynomial approximation."""
    s = x - x0
    n = x.shape[0]

    if len(derivatives) < 2:
        H_diag = np.zeros(n, dtype=s.dtype)
    else:
        H_diag = derivatives[1].copy()

    if len(derivatives) > 2:
        for i, d_diag in enumerate(derivatives[2:], 3):
            H_diag += d_diag * (s**(i - 2)) / math.factorial(i - 2)

    if prox != 0:
        H_diag += prox

    return np.diag(H_diag)

def _poly_minimize(trust_region, prox, de_iters: Any, c, x: torch.Tensor, derivatives):
    derivatives = [T.detach().cpu().numpy().astype(np.float64) for T in derivatives]
    x0 = x.detach().cpu().numpy().astype(np.float64) # taylor series center
    bounds = None
    if trust_region is not None: bounds = list(zip(x0 - trust_region, x0 + trust_region))

    # if len(derivatives) is 1, only gradient is available, I use that to test proximal penalty and bounds
    if bounds is None:
        if len(derivatives) == 1: method = 'bfgs'
        else: method = 'trust-exact'
    else:
        if len(derivatives) == 1: method = 'l-bfgs-b'
        else: method = 'trust-constr'

    x_init = x0.copy()
    v0 = _proximal_poly_v_diag(x0, c, prox, x0, derivatives)
    if de_iters is not None and de_iters != 0:
        if de_iters == -1: de_iters = None # let scipy decide
        res = scipy.optimize.differential_evolution(
            _proximal_poly_v_diag,
            bounds if bounds is not None else list(zip(x0 - 10, x0 + 10)),
            args=(c, prox, x0.copy(), derivatives),
            maxiter=de_iters,
            vectorized=True,
        )
        if res.fun < v0: x_init = res.x

    res = scipy.optimize.minimize(
        _proximal_poly_v_diag,
        x_init,
        method=method,
        args=(c, prox, x0.copy(), derivatives),
        jac=_proximal_poly_g_diag,
        hess=_proximal_poly_H_diag,
        bounds=bounds
    )

    return torch.from_numpy(res.x).to(x), res.fun



class DiagonalHigherOrderNewton(Module):
    """
    Hvp with ones doesn't give you the diagonal unless derivatives are diagonal, but somehow it still works,
    except it doesn't work in all cases except ones where it works.
    """
    def __init__(
        self,
        order: int = 4,
        trust_method: Literal['bounds', 'proximal', 'none'] | None = 'bounds',
        increase: float = 1.5,
        decrease: float = 0.75,
        trust_init: float | None = None,
        trust_tol: float = 1,
        de_iters: int | None = None,
        vectorize: bool = True,
    ):
        if trust_init is None:
            if trust_method == 'bounds': trust_init = 1
            else: trust_init = 0.1

        defaults = dict(order=order, trust_method=trust_method, increase=increase, decrease=decrease, trust_tol=trust_tol, trust_init=trust_init, vectorize=vectorize, de_iters=de_iters)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)
        closure = var.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        settings = self.settings[params[0]]
        order = settings['order']
        increase = settings['increase']
        decrease = settings['decrease']
        trust_tol = settings['trust_tol']
        trust_init = settings['trust_init']
        trust_method = settings['trust_method']
        de_iters = settings['de_iters']

        trust_value = self.global_state.get('trust_value', trust_init)


        # ------------------------ calculate grad and hessian ------------------------ #
        with torch.enable_grad():
            loss = var.loss = var.loss_approx = closure(False)

            g = torch.autograd.grad(loss, params, create_graph=True)
            var.grad = list(g)

            derivatives = [g]
            T = g # current derivatives tensor diagonal
            ones = [torch.ones_like(t) for t in g]

            # get all derivatives up to order
            for o in range(2, order + 1):
                T = hvp(params, T, ones, create_graph=o != order)
                derivatives.append(T)

        x0 = torch.cat([p.ravel() for p in params])

        if trust_method is None: trust_method = 'none'
        else: trust_method = trust_method.lower()

        if trust_method == 'none':
            trust_region = None
            prox = 0

        elif trust_method == 'bounds':
            trust_region = trust_value
            prox = 0

        elif trust_method == 'proximal':
            trust_region = None
            prox = 1 / trust_value

        else:
            raise ValueError(trust_method)

        x_star, expected_loss = _poly_minimize(
            trust_region=trust_region,
            prox=prox,
            de_iters=de_iters,
            c=loss.item(),
            x=x0,
            derivatives=[torch.cat([t.ravel() for t in d]) for d in derivatives],
        )

        # trust region
        if trust_method != 'none':
            expected_reduction = loss - expected_loss

            vec_to_tensors_(x_star, params)
            loss_star = closure(False)
            vec_to_tensors_(x0, params)
            reduction = loss - loss_star

            # failed step
            if reduction <= 0:
                x_star = x0
                self.global_state['trust_value'] = trust_value * decrease

            # very good step
            elif expected_reduction / reduction <= trust_tol:
                self.global_state['trust_value'] = trust_value * increase

        difference = vec_to_tensors(x0 - x_star, params)
        var.update = list(difference)
        return var

