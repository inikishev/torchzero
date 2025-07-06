import itertools
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from typing import Literal

import torch

from ...core import Chainable, Module, apply_transform
from ...utils import TensorList, vec_to_tensors
from ...utils.derivatives import (
    hessian_list_to_mat,
    jacobian_wrt,
)
from ..second_order.newton import (
    cholesky_solve,
    eigh_solve,
    least_squares_solve,
    lu_solve,
)


class NewtonNewton(Module):
    """Applies Newton-like preconditioning to Newton step.

    This is a method that I thought of and then it worked. Here is how it works:

    1. Calculate newton step by solving Hx=g

    2. Calculate jacobian of x wrt parameters and call it H2

    3. Solve H2 x2 = x for x2.

    4. Optionally, repeat (if order is higher than 3.)

    Memory is n^order. It tends to converge faster on convex functions, but can be unstable on non-convex. Orders higher than 3 are usually too unsable and have little benefit.

    3rd order variant can minimize some convex functions with up to 100 variables in less time than Newton's method,
    this is if pytorch can vectorize hessian computation efficiently.
    """
    def __init__(
        self,
        reg: float = 1e-6,
        order: int = 3,
        search_negative: bool = False,
        vectorize: bool = True,
        eigval_tfm: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        defaults = dict(order=order, reg=reg, vectorize=vectorize, eigval_tfm=eigval_tfm, search_negative=search_negative)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)
        closure = var.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        settings = self.settings[params[0]]
        reg = settings['reg']
        vectorize = settings['vectorize']
        order = settings['order']
        search_negative = settings['search_negative']
        eigval_tfm = settings['eigval_tfm']

        # ------------------------ calculate grad and hessian ------------------------ #
        with torch.enable_grad():
            loss = var.loss = var.loss_approx = closure(False)
            g_list = torch.autograd.grad(loss, params, create_graph=True)
            var.grad = list(g_list)

            xp = torch.cat([t.ravel() for t in g_list])
            I = torch.eye(xp.numel(), dtype=xp.dtype, device=xp.device)

            for o in range(2, order + 1):
                is_last = o == order
                H_list = jacobian_wrt([xp], params, create_graph=not is_last, batched=vectorize)
                with torch.no_grad() if is_last else nullcontext():
                    H = hessian_list_to_mat(H_list)
                    if reg != 0: H = H + I * reg

                    x = None
                    if search_negative or (is_last and eigval_tfm is not None):
                        x = eigh_solve(H, xp, eigval_tfm, search_negative=search_negative)
                    if x is None: x = cholesky_solve(H, xp)
                    if x is None: x = lu_solve(H, xp)
                    if x is None: x = least_squares_solve(H, xp)
                    xp = x.squeeze()

        var.update = vec_to_tensors(xp.nan_to_num_(0,0,0), params)
        return var

