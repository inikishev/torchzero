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
    """A method I thought of."""
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



class AcceleratedNewtonNewton(Module):
    """A method I thought of."""
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

            x_list = []
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
                    x_list.append(xp)
                    if len(x_list) >= 2:
                        x1, x2 = x_list[-2:]
                        d = (3/2)*x2 - (1/2)*x1
                        x_list[-1] = d
                        xp = d
        vars.update = vec_to_tensors(xp, params)
        return vars

# class NewtonNewton2(Module):
#     """A method I thought of, except it is another version of it."""
#     def __init__(
#         self,
#         reg: float = 1e-6,
#         order: int = 3,
#         vectorize: bool = True,
#     ):
#         defaults = dict(order=order, reg=reg, vectorize=vectorize)
#         super().__init__(defaults)

#     @torch.no_grad
#     def step(self, vars):
#         params = TensorList(vars.params)
#         closure = vars.closure
#         if closure is None: raise RuntimeError('NewtonCG requires closure')

#         settings = self.settings[params[0]]
#         reg = settings['reg']
#         vectorize = settings['vectorize']
#         order = settings['order']

#         # ------------------------ calculate grad and hessian ------------------------ #
#         with torch.enable_grad():
#             loss = vars.loss = vars.loss_approx = closure(False)
#             g_list = torch.autograd.grad(loss, params, create_graph=True)
#             vars.grad = list(g_list)

#             g = torch.cat([t.view(-1) for t in g_list])
#             I = torch.eye(g.numel(), dtype=g.dtype, device=g.device)

#             xp = g
#             hessians = []
#             for o in range(2, order + 1):
#                 is_last = o == order
#                 print(f'{xp.shape = }')
#                 H_list = jacobian_wrt([xp], params, create_graph=not is_last, batched=vectorize)
#                 print(f'{[a.shape for a in H_list] = }')
#                 with torch.no_grad() if is_last else nullcontext():
#                     H = hessian_list_to_mat(H_list)
#                     if reg != 0:
#                         I = torch.eye(H.size(-1), dtype=H.dtype, device=H.device)
#                         print(f'{H.shape = }')
#                         print(f'{I.shape = }')
#                         H = H + I * reg
#                     hessians.append(H)
#                     xp = H.view(-1)

#         def solve(A, b):
#             x = cholesky_solve(A, b)
#             if x is None: x = lu_solve(A, b)
#             if x is None: x = least_squares_solve(A, b)
#             return x.squeeze()

#         hessians.reverse()
#         for H2, H1 in itertools.pairwise(hessians):
#             H1.set_(solve(H2, H1.view(-1)).view_as(H1))

#         H = hessians[-1]
#         x = solve(H, g)

#         vars.update = vec_to_tensors(x, params)
#         return vars


