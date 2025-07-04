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
    """A basic arbitrary order newton's method with optional trust region and proximal penalty.

    This constructs an nth order taylor approximation via autograd and minimizes it with
    scipy.optimize.minimize trust region newton solvers with optional proximal penalty.

    The trust region currently uses box bounds (L-inf norm), because I don't know how to do L2 for this,


    .. note::
        In most cases HigherOrderNewton should be the first module in the chain because it relies on extra autograd. Use the :code:`inner` argument if you wish to apply Newton preconditioning to another module's output.

    .. note::
        This module requires the a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients for calculating higher order derivatives.
        The closure must accept a ``backward`` argument (refer to documentation).

    .. warning::
        this uses roughly O(N^order) memory and solving the subproblem can be very expensive.

    .. warning::
        "none" and "proximal" trust methods may generate subproblems that have no minima, causing divergence.

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
        increase: float = 2,
        decrease: float = 0.25,
        trust_init: float | None = None,
        eta: float = 1e-6,
        de_iters: int | None = None,
        vectorize: bool = True,
    ):
        if trust_init is None:
            if trust_method == 'bounds': trust_init = 1
            else: trust_init = 0.1

        defaults = dict(order=order, trust_method=trust_method, increase=increase, decrease=decrease, eta=eta, trust_init=trust_init, vectorize=vectorize, de_iters=de_iters)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)
        closure = var.closure
        if closure is None: raise RuntimeError('HigherOrderNewton requires closure')

        settings = self.settings[params[0]]
        order = settings['order']
        increase = settings['increase']
        decrease = settings['decrease']
        eta = settings['eta']
        trust_init = settings['trust_init']
        trust_method = settings['trust_method']
        de_iters = settings['de_iters']
        vectorize = settings['vectorize']

        trust_value = self.global_state.get('trust_region', trust_init)
        if trust_value < 1e-8: trust_region = self.global_state['trust_region'] = settings['init']


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
        success = False
        if trust_method != 'none':
            pred_reduction = loss - expected_loss

            vec_to_tensors_(x_star, params)
            loss_star = closure(False)
            vec_to_tensors_(x0, params)
            reduction = loss - loss_star

            rho = reduction / (max(pred_reduction, 1e-8))
            # failed step
            if rho < 0.25:
                self.global_state['trust_region'] = trust_value * decrease

            # very good step
            elif rho > 0.75:
                diff = trust_value - (x0 - x_star).abs_()
                if (diff.amin() / trust_value) > 1e-4: # hits boundary
                    self.global_state['trust_region'] = trust_value * increase

            # if the ratio is high enough then accept the proposed step
            success = rho > eta

        if success:
            difference = vec_to_tensors(x0 - x_star, params)
            var.update = list(difference)
        else:
            var.update = params.zeros_like()
        return var

