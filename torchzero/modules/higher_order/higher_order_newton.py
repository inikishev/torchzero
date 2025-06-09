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
)

_LETTERS = 'abcdefghijklmnopqrstuvwxyz'
def _poly_eval(s: np.ndarray, c, derivatives):
    val = float(c)
    for i,T in enumerate(derivatives, 1):
        s1 = ''.join(_LETTERS[:i]) # abcd
        s2 = ',...'.join(_LETTERS[:i]) # a,b,c,d
        # this would make einsum('abcd,a,b,c,d', T, x, x, x, x)
        val += np.einsum(f"...{s1},...{s2}", T, *(s for _ in range(i))) / math.factorial(i)
    return val

def _proximal_poly_v(x: np.ndarray, c, prox, x0: np.ndarray, derivatives):
    if x.ndim == 2: x = x.T # DE passes (ndim, batch_size)
    s = x - x0
    val = _poly_eval(s, c, derivatives)
    penalty = 0
    if prox != 0: penalty = (prox / 2) * (s**2).sum(-1) # proximal penalty
    return val + penalty

def _proximal_poly_g(x: np.ndarray, c, prox, x0: np.ndarray, derivatives):
    s = x - x0
    g = derivatives[0].copy()
    if len(derivatives) > 1:
        for i, T in enumerate(derivatives[1:], 2):
            s1 = ''.join(_LETTERS[:i]) # abcd
            s2 = ','.join(_LETTERS[1:i]) # b,c,d
            # this would make einsum('abcd,b,c,d->a', T, x, x, x)
            g += np.einsum(f"{s1},{s2}->a", T, *(s for _ in range(i-1))) / math.factorial(i - 1)

    g_prox = 0
    if prox != 0: g_prox = prox * s
    return g + g_prox

def _proximal_poly_H(x: np.ndarray, c, prox, x0: np.ndarray, derivatives):
    s = x - x0
    n = x.shape[0]
    if len(derivatives) == 1:
        H = np.zeros(n, n)
    else:
        H = derivatives[1].copy()
        if len(derivatives) > 2:
            for i, T in enumerate(derivatives[2:], 3):
                s1 = ''.join(_LETTERS[:i]) # abcd
                s2 = ','.join(_LETTERS[2:i]) # c,d
                # this would make einsum('abcd,c,d->ab', T, x, x, x)
                H += np.einsum(f"{s1},{s2}->ab", T, *(s for _ in range(i-2))) / math.factorial(i - 2)

    H_prox = 0
    if prox != 0: H_prox = np.eye(n) * prox
    return H + H_prox

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
    v0 = _proximal_poly_v(x0, c, prox, x0, derivatives)
    if de_iters is not None and de_iters != 0:
        if de_iters == -1: de_iters = None # let scipy decide
        res = scipy.optimize.differential_evolution(
            _proximal_poly_v,
            bounds if bounds is not None else list(zip(x0 - 10, x0 + 10)),
            args=(c, prox, x0.copy(), derivatives),
            maxiter=de_iters,
            vectorized=True,
        )
        if res.fun < v0: x_init = res.x

    res = scipy.optimize.minimize(
        _proximal_poly_v,
        x_init,
        method=method,
        args=(c, prox, x0.copy(), derivatives),
        jac=_proximal_poly_g,
        hess=_proximal_poly_H,
        bounds=bounds
    )

    return torch.from_numpy(res.x).to(x), res.fun



class HigherOrderNewton(Module):
    """
    A basic arbitrary order newton's method with optional trust region and proximal penalty.
    It is recommended to enable at least one of trust region or proximal penalty.

    This constructs an nth order taylor approximation via autograd and minimizes it with
    scipy.optimize.minimize trust region newton solvers with optional proximal penalty.

    This uses n^order memory, where n is number of decision variables, and I am not aware
    of any problems where this is more efficient than newton's method. It can minimize
    rosenbrock in a single step, but that step probably takes more time than newton.
    And there are way more efficient tensor methods out there but they tend to be
    significantly more complex.

    Args:

        order (int, optional):
            Order of the method, number of taylor series terms (orders of derivatives) used to approximate the function. Defaults to 4.
        trust_method (str | None, optional):
            Method used for trust region.
            - "bounds" - the model is minimized within bounds defined by trust region.
            - "proximal" - the model is minimized with penalty for going too far from current point.
            - "none" - disables trust region.

            Defaults to 'bounds'.
        increase (float, optional): trust region multiplier on good steps. Defaults to 1.5.
        decrease (float, optional): trust region multiplier on bad steps. Defaults to 0.75.
        trust_init (float | None, optional):
            initial trust region size. If none, defaults to 1 on :code:`trust_method="bounds"` and 0.1 on :code:`"proximal"`. Defaults to None.
        trust_tol (float, optional):
            Maximum ratio of expected loss reduction to actual reduction for trust region increase.
            Should 1 or higer. Defaults to 2.
        de_iters (int | None, optional):
            If this is specified, the model is minimized via differential evolution first to possibly escape local minima,
            then it is passed to scipy.optimize.minimize. Defaults to None.
        vectorize (bool, optional): whether to enable vectorized jacobians (usually faster). Defaults to True.
    """
    def __init__(
        self,
        order: int = 4,
        trust_method: Literal['bounds', 'proximal', 'none'] | None = 'bounds',
        increase: float = 1.5,
        decrease: float = 0.75,
        trust_init: float | None = None,
        trust_tol: float = 2,
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
        vectorize = settings['vectorize']

        trust_value = self.global_state.get('trust_value', trust_init)


        # ------------------------ calculate grad and hessian ------------------------ #
        with torch.enable_grad():
            loss = var.loss = var.loss_approx = closure(False)

            g_list = torch.autograd.grad(loss, params, create_graph=True)
            var.grad = list(g_list)

            g = torch.cat([t.ravel() for t in g_list])
            n = g.numel()
            derivatives = [g]
            T = g # current derivatives tensor

            # get all derivative up to order
            for o in range(2, order + 1):
                is_last = o == order
                T_list = jacobian_wrt([T], params, create_graph=not is_last, batched=vectorize)
                with torch.no_grad() if is_last else nullcontext():
                    # the shape is (ndim, ) * order
                    T = hessian_list_to_mat(T_list).view(n, n, *T.shape[1:])
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
            derivatives=derivatives,
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

