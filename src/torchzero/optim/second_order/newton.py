from typing import Literal
from collections import abc

import torch

from ...core import OptimizerModule, TensorListOptimizer
from ...modules import LR, ClipNorm
from ...modules import ExactNewton as _ExactNewton
from ...modules import (FallbackLinearSystemSolvers, LinearSystemSolvers,
                        LineSearches, get_line_search)
from ..modular import Modular

class ExactNewton(Modular):
    """Peforms an exact Newton step using batched autograd.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. Defaults to 1.
        tikhonov (float, optional):
            tikhonov regularization (constant value added to the diagonal of the hessian). Defaults to 0.
        solver (LinearSystemSolvers, optional):
            solver for Hx = g. Defaults to "cholesky_lu" (cholesky or LU if it fails).
        fallback (FallbackLinearSystemSolvers, optional):
            what to do if solver fails. Defaults to "safe_diag"
            (takes nonzero diagonal elements, or fallbacks to gradient descent if all elements are 0).
        max_norm (float, optional):
            clips the newton step to L2 norm to avoid instability by giant steps.
            A mauch better way is to use trust region methods. I haven't implemented any
            but you can use `tz.optim.wrappers.scipy.ScipyMinimize` with one of the trust region methods.
            Defaults to None.
        validate (bool, optional):
            validate if the step didn't increase the loss by `loss * tol` with an additional forward pass.
            If not, undo the step and perform a gradient descent step.
        tol (float, optional):
            only has effect if `validate` is enabled.
            If loss increased by `loss * tol`, perform gradient descent step.
            Set this to 0 to guarantee that loss always decreases. Defaults to 1.
        gd_lr (float, optional):
            only has effect if `validate` is enabled.
            Gradient descent step learning rate. Defaults to 1e-2.
        line_search (OptimizerModule | None, optional): line search module, can be None. Defaults to None.
        batched_hessian (bool, optional):
            whether to use experimental pytorch vmap-vectorized hessian calculation. As per pytorch docs,
            should be faster, but this feature being experimental, there may be performance cliffs.
            Defaults to True.
        diag (False, optional):
            only use the diagonal of the hessian. This will still calculate the full hessian!
            This is mainly useful for benchmarking.
    """
    def __init__(
        self,
        params,
        lr: float = 1,
        tikhonov: float | Literal['eig'] = 0.0,
        solver: LinearSystemSolvers = "cholesky_lu",
        fallback: FallbackLinearSystemSolvers = "safe_diag",
        max_norm: float | None = None,
        validate=False,
        tol: float = 1,
        gd_lr = 1e-2,
        line_search: LineSearches | None = None,
        batched_hessian = True,

        diag: bool = False,
    ):
        modules: list[OptimizerModule] = [
            _ExactNewton(
                tikhonov=tikhonov,
                batched_hessian=batched_hessian,
                solver=solver,
                fallback=fallback,
                validate=validate,
                tol = tol,
                gd_lr=gd_lr,
                diag = diag,
            ),
        ]

        if max_norm is not None:
            modules.append(ClipNorm(max_norm))

        if lr != 1:
            modules.append(LR(lr))

        if line_search is not None:
            modules.append(get_line_search(line_search))

        super().__init__(params, modules)


