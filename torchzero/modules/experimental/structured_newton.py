# idea https://arxiv.org/pdf/2212.09841
import warnings
from collections.abc import Callable
from functools import partial
from typing import Literal

import torch

from ...core import Chainable, Module, apply_transform
from ...utils import TensorList, vec_to_tensors
from ...utils.derivatives import (
    hessian_list_to_mat,
    hessian_mat,
    hvp,
    hvp_fd_central,
    hvp_fd_forward,
    jacobian_and_hessian_wrt,
)


class StructuredNewton(Module):
    """TODO. Please note that this is experimental and isn't guaranteed to work.
    Args:
        structure (str, optional): structure.
        reg (float, optional): tikhonov regularizer value. Defaults to 1e-6.
        hvp_method (str):
            how to calculate hvp_method. Defaults to "autograd".
        inner (Chainable | None, optional): inner modules. Defaults to None.

    """
    def __init__(
        self,
        structure: Literal[
            "diagonal",
            "diagonal1",
            "diagonal_abs",
            "tridiagonal",
            "circulant",
            "toeplitz",
            "toeplitz_like",
            "hankel",
            "rank1",
            "rank2", # any rank
        ]
        | str = "diagonal",
        reg: float = 1e-6,
        hvp_method: Literal["autograd", "forward", "central"] = "autograd",
        h: float = 1e-3,
        inner: Chainable | None = None,
    ):
        defaults = dict(reg=reg, hvp_method=hvp_method, structure=structure, h=h)
        super().__init__(defaults)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)
        closure = var.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        settings = self.settings[params[0]]
        reg = settings['reg']
        hvp_method = settings['hvp_method']
        structure = settings['structure']
        h = settings['h']

        # ------------------------ calculate grad and hessian ------------------------ #
        if hvp_method == 'autograd':
            grad = var.get_grad(create_graph=True)
            def Hvp_fn1(x):
                return hvp(params, grad, x, retain_graph=True)
            Hvp_fn = Hvp_fn1

        elif hvp_method == 'forward':
            grad = var.get_grad()
            def Hvp_fn2(x):
                return hvp_fd_forward(closure, params, x, h=h, g_0=grad, normalize=True)[1]
            Hvp_fn = Hvp_fn2

        elif hvp_method == 'central':
            grad = var.get_grad()
            def Hvp_fn3(x):
                return hvp_fd_central(closure, params, x, h=h, normalize=True)[1]
            Hvp_fn = Hvp_fn3

        else: raise ValueError(hvp_method)

        # -------------------------------- inner step -------------------------------- #
        update = var.get_update()
        if 'inner' in self.children:
            update = apply_transform(self.children['inner'], update, params=params, grads=grad, var=var)

        # hessian
        if structure.startswith('diagonal'):
            H = Hvp_fn([torch.ones_like(p) for p in params])
            if structure == 'diagonal1': torch._foreach_clamp_min_(H, 1)
            if structure == 'diagonal_abs': torch._foreach_abs_(H)
            torch._foreach_add_(H, reg)
            torch._foreach_div_(update, H)
            var.update = update
            return var

        # hessian
        raise NotImplementedError(structure)





