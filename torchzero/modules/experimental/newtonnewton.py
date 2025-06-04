from contextlib import nullcontext
import warnings
from collections.abc import Callable
from functools import partial
import itertools
from typing import Literal

import torch

from ...core import Chainable, Module, apply
from ...utils import TensorList, vec_to_tensors
from ...utils.derivatives import (
    hessian_list_to_mat,
    jacobian_wrt,
)


def lu_solve(H: torch.Tensor, g: torch.Tensor):
    x, info = torch.linalg.solve_ex(H, g) # pylint:disable=not-callable
    if info == 0: return x
    return None

def cholesky_solve(H: torch.Tensor, g: torch.Tensor):
    x, info = torch.linalg.cholesky_ex(H) # pylint:disable=not-callable
    if info == 0:
        return torch.cholesky_solve(g.unsqueeze(1), x)
    return None

def least_squares_solve(H: torch.Tensor, g: torch.Tensor):
    return torch.linalg.lstsq(H, g)[0] # pylint:disable=not-callable

def eigh_solve(H: torch.Tensor, g: torch.Tensor, tfm: Callable | None):
    try:
        L, Q = torch.linalg.eigh(H) # pylint:disable=not-callable
        if tfm is not None: L = tfm(L)
        L = L.reciprocal()
        return torch.linalg.multi_dot([Q * L.unsqueeze(-2), Q.mH, g]) # pylint:disable=not-callable
    except torch.linalg.LinAlgError:
        return None

class NewtonNewton(Module):
    """
    Method that I thought of and then it worked.

    1. Calculate newton step by solving Hx=g

    2. Calculate jacobian of x wrt parameters and call it H2

    3. Solve H2 x2 = x for x2.

    4. Optionally, repeat (if order is higher than 3.)

    Memory is n^order. It tends to converge faster on convex functions, but can be unstable on non-convex. Orders higher than 3 are usually too unsable and have little benefit.
    """
    def __init__(
        self,
        reg: float = 1e-6,
        order: int = 3,
        vectorize: bool = True,
        eigval_tfm: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        defaults = dict(order=order, reg=reg, vectorize=vectorize, eigval_tfm=eigval_tfm)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, vars):
        params = TensorList(vars.params)
        closure = vars.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        settings = self.settings[params[0]]
        reg = settings['reg']
        vectorize = settings['vectorize']
        order = settings['order']
        eigval_tfm = settings['eigval_tfm']

        # ------------------------ calculate grad and hessian ------------------------ #
        with torch.enable_grad():
            loss = vars.loss = vars.loss_approx = closure(False)
            g_list = torch.autograd.grad(loss, params, create_graph=True)
            vars.grad = list(g_list)

            xp = torch.cat([t.ravel() for t in g_list])
            I = torch.eye(xp.numel(), dtype=xp.dtype, device=xp.device)

            for o in range(2, order + 1):
                is_last = o == order
                H_list = jacobian_wrt([xp], params, create_graph=not is_last, batched=vectorize)
                with torch.no_grad() if is_last else nullcontext():
                    H = hessian_list_to_mat(H_list)
                    if reg != 0: H = H + I * reg

                    x = None
                    if is_last and eigval_tfm is not None: x = eigh_solve(H, xp, eigval_tfm)
                    if x is None: x = cholesky_solve(H, xp)
                    if x is None: x = lu_solve(H, xp)
                    if x is None: x = least_squares_solve(H, xp)
                    xp = x.squeeze()

        vars.update = vec_to_tensors(xp, params)
        return vars

