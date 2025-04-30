import warnings
from functools import partial
from typing import Literal

import torch

from ...core import Chainable, apply
from ...utils import vec_to_tensors
from ...utils.derivatives import (
    hessian_list_to_mat,
    hessian_mat,
    jacobian_and_hessian_wrt,
)
from ..grad_approximation import GradMaker, GradTarget


def lu_solve(H: torch.Tensor, g: torch.Tensor):
    x, info = torch.linalg.solve_ex(H, g) # pylint:disable=not-callable
    if info == 0: return x
    return None

def cholesky_solve(H: torch.Tensor, g: torch.Tensor):
    x, info = torch.linalg.cholesky_ex(H) # pylint:disable=not-callable
    if info == 0:
        g.unsqueeze_(1)
        return torch.cholesky_solve(g, x)
    return None

def least_squares_solve(H: torch.Tensor, g: torch.Tensor):
    return torch.linalg.lstsq(H, g)[0] # pylint:disable=not-callable

def tikhonov_(H: torch.Tensor, reg: float):
    if reg!=0: H.add_(torch.eye(H.size(-1), dtype=H.dtype, device=H.device).mul_(reg))
    return H

def eig_tikhonov_(H: torch.Tensor, reg: float):
    v = torch.linalg.eigvalsh(H).min().clamp_(max=0).neg_() + reg # pylint:disable=not-callable
    return tikhonov_(H, v)

def inv_matrix_clamp(H: torch.Tensor, reg: float):
    try:
        eigvals, eigvecs = torch.linalg.eigh(H) # pylint:disable=not-callable
        eigvals.clamp_(min=reg).reciprocal_()
        return eigvecs @ torch.diag(eigvals) @ eigvecs.mH
    except Exception:
        return None


class Newton(GradMaker):
    """Exact newton via autograd.

    Args:
        reg (float, optional): tikhonov regularizer value. Defaults to 1e-6.
        hessian_clamp (float | None, optional):
            if not None, clamp eigenvalues to be larger than this value. This is usually better
            than eig_reg, plus eigenvectors will be reused to invert the hessian. Defaults to None.
        eig_reg (bool, optional): whether to use largest negative eigenvalue as regularizer. Defaults to False.
        hessian_method (str):
            how to calculate hessian. Defaults to "autograd".
        vectorize (bool, optional):
            whether to enable vectorized hessian. Defaults to True.
        target (GradTarget, optional): target. Defaults to "closure".
        inner (Chainable | None, optional): inner modules. Defaults to None.
    """
    def __init__(
        self,
        reg: float = 1e-6,
        hessian_clamp: float | None = None,
        eig_reg: bool = False,
        hessian_method: Literal["autograd", "func", "autograd.functional"] = "autograd",
        vectorize: bool = True,
        target: GradTarget = "update",
        inner: Chainable | None = None,
    ):
        defaults = dict(reg=reg, eig_reg=eig_reg, hessian_clamp=hessian_clamp, hessian_method=hessian_method, vectorize=vectorize)
        super().__init__(defaults, target=target)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def approximate(self, closure, params, loss, vars):
        settings = self.settings[params[0]]
        reg = settings['reg']
        eig_reg = settings['eig_reg']
        hessian_clamp = settings['hessian_clamp']
        hessian_method = settings['hessian_method']
        vectorize = settings['vectorize']

        # ------------------------ calculate grad and hessian ------------------------ #
        if hessian_method == 'autograd':
            with torch.enable_grad():
                loss = closure(False)
                g_list, H_list = jacobian_and_hessian_wrt([loss], params, batched=vectorize)
                g_list = [t[0] for t in g_list] # remove leading dim from loss
                H = hessian_list_to_mat(H_list)

        elif hessian_method in ('func', 'autograd.functional'):
            strat = 'forward-mode' if vectorize else 'reverse-mode'
            with torch.enable_grad():
                vars.zero_grad()
                loss = closure(False)
                loss.backward(retain_graph=True)
                g_list = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                H: torch.Tensor = hessian_mat(partial(closure, backward=False), params,
                                method=hessian_method, vectorize=vectorize, outer_jacobian_strategy=strat) # pyright:ignore[reportAssignmentType]

        else:
            raise ValueError(hessian_method)

        # -------------------------------- inner step -------------------------------- #
        if 'inner' in self.children:
            g_list = apply(self.children['inner'], list(g_list), params=params, grad=list(g_list), vars=vars)
        g = torch.cat([t.view(-1) for t in g_list])

        # ------------------------------- regulazition ------------------------------- #
        if eig_reg: H = eig_tikhonov_(H, reg)
        else: H = tikhonov_(H, reg)

        # ----------------------------------- solve ---------------------------------- #
        update = None
        if hessian_clamp is not None:
            update = inv_matrix_clamp(H, hessian_clamp) @ g

        if update is None: update = cholesky_solve(H, g)
        if update is None: update = lu_solve(H, g)
        if update is None: update = least_squares_solve(H, g)

        return vec_to_tensors(update, params), loss, loss