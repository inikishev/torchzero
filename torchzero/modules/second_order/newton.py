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


def lu_solve(H: torch.Tensor, g: torch.Tensor):
    try:
        x, info = torch.linalg.solve_ex(H, g) # pylint:disable=not-callable
        if info == 0: return x
        return None
    except RuntimeError:
        return None

def cholesky_solve(H: torch.Tensor, g: torch.Tensor):
    x, info = torch.linalg.cholesky_ex(H) # pylint:disable=not-callable
    if info == 0:
        g.unsqueeze_(1)
        return torch.cholesky_solve(g, x)
    return None

def least_squares_solve(H: torch.Tensor, g: torch.Tensor):
    return torch.linalg.lstsq(H, g)[0] # pylint:disable=not-callable

def eigh_solve(H: torch.Tensor, g: torch.Tensor, tfm: Callable | None, search_negative: bool):
    try:
        L, Q = torch.linalg.eigh(H) # pylint:disable=not-callable
        if tfm is not None: L = tfm(L)
        if search_negative and L[0] < 0:
            d = Q[0]
             # use eigvec or -eigvec depending on if it points in same direction as gradient
            return g.dot(d).sign() * d

        return Q @ ((Q.mH @ g) / L)

    except torch.linalg.LinAlgError:
        return None

def tikhonov_(H: torch.Tensor, reg: float):
    if reg!=0: H.add_(torch.eye(H.size(-1), dtype=H.dtype, device=H.device).mul_(reg))
    return H


class Newton(Module):
    """Exact newton's method via autograd.

    .. note::
        In most cases Newton should be the first module in the chain because it relies on autograd. Use the :code:`inner` argument if you wish to apply Newton preconditioning to another module's output.

    .. note::
        This module requires the a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients for calculating the hessian.
        The closure must accept a ``backward`` argument (refer to documentation).

    .. warning::
        this uses roughly O(N^2) memory.


    Args:
        reg (float, optional): tikhonov regularizer value. Defaults to 1e-6.
        search_negative (bool, Optional):
            if True, whenever a negative eigenvalue is detected,
            search direction is proposed along an eigenvector corresponding to a negative eigenvalue.
        hessian_method (str):
            how to calculate hessian. Defaults to "autograd".
        vectorize (bool, optional):
            whether to enable vectorized hessian. Defaults to True.
        inner (Chainable | None, optional): modules to apply hessian preconditioner to. Defaults to None.
        H_tfm (Callable | None, optional):
            optional hessian transforms, takes in two arguments - `(hessian, gradient)`.

            must return either a tuple: `(hessian, is_inverted)` with transformed hessian and a boolean value
            which must be True if transform inverted the hessian and False otherwise.

            Or it returns a single tensor which is used as the update.

            Defaults to None.
        eigval_tfm (Callable | None, optional):
            optional eigenvalues transform, for example :code:`torch.abs` or :code:`lambda L: torch.clip(L, min=1e-8)`.
            If this is specified, eigendecomposition will be used to invert the hessian.

    Examples:
        Newton's method with backtracking line search

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.Newton(),
                tz.m.Backtracking()
            )

        Newton's method modified for non-convex functions by taking matrix absolute value of the hessian

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.Newton(eigval_tfm=lambda x: torch.abs(x).clip(min=0.1)),
                tz.m.Backtracking()
            )

        Newton's method modified for non-convex functions by searching along negative curvature directions

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.Newton(search_negative=True),
                tz.m.Backtracking()
            )

        Newton preconditioning applied to momentum

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.Newton(inner=tz.m.EMA(0.9)),
                tz.m.LR(0.1)
            )

        Diagonal newton example. This will still evaluate the entire hessian so it isn't efficient, but if you wanted to see how diagonal newton behaves or compares to full newton, you can use this.

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.Newton(H_tfm = lambda H, g: g/H.diag()),
                tz.m.Backtracking()
            )

    """
    def __init__(
        self,
        reg: float = 1e-6,
        search_negative: bool = False,
        update_freq: int = 1,
        hessian_method: Literal["autograd", "func", "autograd.functional"] = "autograd",
        vectorize: bool = True,
        inner: Chainable | None = None,
        H_tfm: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, bool]] | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        eigval_tfm: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        defaults = dict(reg=reg, hessian_method=hessian_method, vectorize=vectorize, H_tfm=H_tfm, eigval_tfm=eigval_tfm, search_negative=search_negative, update_freq=update_freq)
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
        search_negative = settings['search_negative']
        hessian_method = settings['hessian_method']
        vectorize = settings['vectorize']
        H_tfm = settings['H_tfm']
        eigval_tfm = settings['eigval_tfm']
        update_freq = settings['update_freq']

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        g_list = var.grad
        H = None
        if step % update_freq == 0:
            # ------------------------ calculate grad and hessian ------------------------ #
            if hessian_method == 'autograd':
                with torch.enable_grad():
                    loss = var.loss = var.loss_approx = closure(False)
                    g_list, H_list = jacobian_and_hessian_wrt([loss], params, batched=vectorize)
                    g_list = [t[0] for t in g_list] # remove leading dim from loss
                    var.grad = g_list
                    H = hessian_list_to_mat(H_list)

            elif hessian_method in ('func', 'autograd.functional'):
                strat = 'forward-mode' if vectorize else 'reverse-mode'
                with torch.enable_grad():
                    g_list = var.get_grad(retain_graph=True)
                    H = hessian_mat(partial(closure, backward=False), params,
                                    method=hessian_method, vectorize=vectorize, outer_jacobian_strategy=strat) # pyright:ignore[reportAssignmentType]

            else:
                raise ValueError(hessian_method)

            H = tikhonov_(H, reg)
            if update_freq != 1:
                self.global_state['H'] = H

        if H is None:
            H = self.global_state["H"]

        # var.storage['hessian'] = H

        # -------------------------------- inner step -------------------------------- #
        update = var.get_update()
        if 'inner' in self.children:
            update = apply_transform(self.children['inner'], update, params=params, grads=g_list, var=var)

        g = torch.cat([t.ravel() for t in update])


        # ----------------------------------- solve ---------------------------------- #
        update = None
        if H_tfm is not None:
            ret = H_tfm(H, g)

            if isinstance(ret, torch.Tensor):
                update = ret

            else: # returns (H, is_inv)
                H, is_inv = ret
                if is_inv: update = H @ g

        if search_negative or (eigval_tfm is not None):
            update = eigh_solve(H, g, eigval_tfm, search_negative=search_negative)

        if update is None: update = cholesky_solve(H, g)
        if update is None: update = lu_solve(H, g)
        if update is None: update = least_squares_solve(H, g)

        var.update = vec_to_tensors(update, params)

        return var

class InverseFreeNewton(Module):
    """Inverse-free newton's method

    .. note::
        In most cases Newton should be the first module in the chain because it relies on autograd. Use the :code:`inner` argument if you wish to apply Newton preconditioning to another module's output.

    .. note::
        This module requires the a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients for calculating the hessian.
        The closure must accept a ``backward`` argument (refer to documentation).

    .. warning::
        this uses roughly O(N^2) memory.

    Reference
        Massalski, Marcin, and Magdalena Nockowska-Rosiak. "INVERSE-FREE NEWTON'S METHOD." Journal of Applied Analysis & Computation 15.4 (2025): 2238-2257.
    """
    def __init__(
        self,
        update_freq: int = 1,
        hessian_method: Literal["autograd", "func", "autograd.functional"] = "autograd",
        vectorize: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(hessian_method=hessian_method, vectorize=vectorize, update_freq=update_freq)
        super().__init__(defaults)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)
        closure = var.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        settings = self.settings[params[0]]
        hessian_method = settings['hessian_method']
        vectorize = settings['vectorize']
        update_freq = settings['update_freq']

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        g_list = var.grad
        Y = None
        if step % update_freq == 0:
            # ------------------------ calculate grad and hessian ------------------------ #
            if hessian_method == 'autograd':
                with torch.enable_grad():
                    loss = var.loss = var.loss_approx = closure(False)
                    g_list, H_list = jacobian_and_hessian_wrt([loss], params, batched=vectorize)
                    g_list = [t[0] for t in g_list] # remove leading dim from loss
                    var.grad = g_list
                    H = hessian_list_to_mat(H_list)

            elif hessian_method in ('func', 'autograd.functional'):
                strat = 'forward-mode' if vectorize else 'reverse-mode'
                with torch.enable_grad():
                    g_list = var.get_grad(retain_graph=True)
                    H = hessian_mat(partial(closure, backward=False), params,
                                    method=hessian_method, vectorize=vectorize, outer_jacobian_strategy=strat) # pyright:ignore[reportAssignmentType]

            else:
                raise ValueError(hessian_method)

            # inverse free part
            if 'Y' not in self.global_state:
                num = H.T
                denom = (torch.linalg.norm(H, 1) * torch.linalg.norm(H, float('inf'))) # pylint:disable=not-callable
                eps = torch.finfo(H.dtype).eps
                Y = self.global_state['Y'] = num.div_(denom.clip(min=eps, max=1/eps))

            else:
                Y = self.global_state['Y']
                I = torch.eye(Y.size(0), device=Y.device, dtype=Y.dtype).mul_(2)
                I -= H @ Y
                Y = self.global_state['Y'] = Y @ I

        if Y is None:
            Y = self.global_state["Y"]


        # -------------------------------- inner step -------------------------------- #
        update = var.get_update()
        if 'inner' in self.children:
            update = apply_transform(self.children['inner'], update, params=params, grads=g_list, var=var)

        g = torch.cat([t.ravel() for t in update])


        # ----------------------------------- solve ---------------------------------- #
        var.update = vec_to_tensors(Y@g, params)

        return var
