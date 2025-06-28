from contextlib import nullcontext
import warnings
from collections.abc import Callable
from functools import partial
import itertools
from typing import Literal

import torch

from ...core import Chainable, Module, apply_transform
from ...utils import TensorList, vec_to_tensors
from ...utils.derivatives import (
    hessian_list_to_mat,
    jacobian_wrt, jacobian_and_hessian_wrt, hessian_mat,
)

def _batched_dot(x, y):
    return (x.unsqueeze(-2) @ y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

def _cosine_similarity(x, y):
    denom = torch.linalg.vector_norm(x, dim=-1) * torch.linalg.vector_norm(y, dim=-1).clip(min=torch.finfo(x.dtype).eps) # pylint:disable=not-callable
    return _batched_dot(x, y) / denom

class EigenDescent(Module):
    """
    Uses eigenvectors corresponding to certain eigenvalues. For now they are just extracted from hessian.

    .. warning::
        Experimental.

    Args:
        mode (str, optional):
            - largest - use largest eigenvalue unless all eigenvalues are negative, then smallest is used.
            - smallest - use smallest eigenvalue unless all eigenvalues are positive, then largest is used.
            - mean-sign - use mean of eigenvectors multiplied by 1 or -1 if they point in opposite direction from gradient.
            - mean-dot - use mean of eigenvectors multiplied by dot product with gradient.
            - mean-cosine - use mean of eigenvectors multiplied by cosine similarity with gradient.
            - mm - for testing.

            Defaults to 'mean-sign'.
        hessian_method (str, optional): how to calculate hessian. Defaults to "autograd".
        vectorize (bool, optional): how to calculate hessian. Defaults to True.

    """
    def __init__(
        self,
        mode: Literal['largest', 'smallest','magnitude', 'mean-sign', 'mean-dot', 'mean-cosine', 'mm'] = 'mean-sign',
        hessian_method: Literal["autograd", "func", "autograd.functional"] = "autograd",
        vectorize: bool = True,
    ):
        defaults = dict(hessian_method=hessian_method, vectorize=vectorize, mode=mode)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)
        closure = var.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        settings = self.settings[params[0]]
        mode = settings['mode']
        hessian_method = settings['hessian_method']
        vectorize = settings['vectorize']

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
                H: torch.Tensor = hessian_mat(partial(closure, backward=False), params,
                                method=hessian_method, vectorize=vectorize, outer_jacobian_strategy=strat) # pyright:ignore[reportAssignmentType]

        else:
            raise ValueError(hessian_method)


        # ----------------------------------- solve ---------------------------------- #
        g = torch.cat([t.ravel() for t in g_list])
        L, Q = torch.linalg.eigh(H) # L is sorted # pylint:disable=not-callable
        if mode == 'largest':
            # smallest eigenvalue if all eigenvalues are negative else largest
            if L[-1] <= 0: d = Q[0]
            else: d = Q[-1]

        elif mode == 'smallest':
            # smallest eigenvalue if negative eigenvalues exist else largest
            if L[0] <= 0: d = Q[0]
            else: d = Q[-1]

        elif mode == 'magnitude':
            # largest by magnitude
            if L[0].abs() > L[-1].abs(): d = Q[0]
            else: d = Q[-1]

        elif mode == 'mean-dot':
            d = ((g.unsqueeze(0) @ Q).squeeze(0) * Q).mean(1)

        elif mode == 'mean-sign':
            d = ((g.unsqueeze(0) @ Q).squeeze(0).sign() * Q).mean(1)

        elif mode == 'mean-cosine':
            d = (Q * _cosine_similarity(Q, g)).mean(1)

        elif mode == 'mm':
            d = (g.unsqueeze(0) @ Q).squeeze(0) / g.numel()

        else:
            raise ValueError(mode)

        var.update = vec_to_tensors(g.dot(d).sign() * d, params)
        return var

