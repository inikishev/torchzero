from typing import Any, Literal
import torch

from ...modules import (LR, FallbackLinearSystemSolvers,
                        LinearSystemSolvers, LineSearches, ClipNorm)
from ...modules import NewtonFDM as _NewtonFDM, get_line_search
from ...modules.experimental.subspace import Proj2Masks, ProjRandom, Subspace
from ..modular import Modular


class NewtonFDM(Modular):
    """Newton method with gradient and hessian approximated via finite difference.

    This performs approximately `4 * n^2 + 1` evaluations per step;
    if `diag` is True, performs `n * 2 + 1` evaluations per step.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate.
        eps (float, optional): epsilon for finite difference.
            Note that with float32 this needs to be quite high to avoid numerical instability. Defaults to 1e-2.
        diag (bool, optional): whether to only approximate diagonal elements of the hessian.
            This also ignores `solver` if True. Defaults to False.
        solver (LinearSystemSolvers, optional):
            solver for Hx = g. Defaults to "cholesky_lu" (cholesky or LU if it fails).
        fallback (FallbackLinearSystemSolvers, optional):
            what to do if solver fails. Defaults to "safe_diag"
            (takes nonzero diagonal elements, or fallbacks to gradient descent if all elements are 0).
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
        line_search (OptimizerModule | None, optional): line search module, can be None. Defaults to 'brent'.
    """
    def __init__(
        self,
        params,
        lr: float = 1,
        eps: float = 1e-2,
        diag=False,
        solver: LinearSystemSolvers = "cholesky_lu",
        fallback: FallbackLinearSystemSolvers = "safe_diag",
        max_norm: float | None = None,
        validate=False,
        tol: float = 2,
        gd_lr = 1e-2,
        line_search: LineSearches | None = 'brent',
    ):
        modules: list[Any] = [
            _NewtonFDM(eps = eps, diag = diag, solver=solver, fallback=fallback, validate=validate, tol=tol, gd_lr=gd_lr),
        ]

        if max_norm is not None:
            modules.append(ClipNorm(max_norm))

        modules.append(LR(lr))

        if line_search is not None:
            modules.append(get_line_search(line_search))

        super().__init__(params, modules)


class RandomSubspaceNewtonFDM(Modular):
    """This projects the parameters into a smaller dimensional subspace,
    making approximating the hessian via finite difference feasible.

    This performs approximately `4 * subspace_ndim^2 + 1` evaluations per step;
    if `diag` is True, performs `subspace_ndim * 2 + 1` evaluations per step.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        subspace_ndim (float, optional): number of random subspace dimensions.
        lr (float, optional): learning rate.
        eps (float, optional): epsilon for finite difference.
            Note that with float32 this needs to be quite high to avoid numerical instability. Defaults to 1e-2.
        diag (bool, optional): whether to only approximate diagonal elements of the hessian.
        solver (LinearSystemSolvers, optional):
            solver for Hx = g. Defaults to "cholesky_lu" (cholesky or LU if it fails).
        fallback (FallbackLinearSystemSolvers, optional):
            what to do if solver fails. Defaults to "safe_diag"
            (takes nonzero diagonal elements, or fallbacks to gradient descent if all elements are 0).
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
        line_search (OptimizerModule | None, optional): line search module, can be None. Defaults to BacktrackingLS().
        randomize_every (float, optional): generates new random projections every n steps. Defaults to 1.
    """
    def __init__(
        self,
        params,
        subspace_ndim: int = 3,
        lr: float = 1,
        eps: float = 1e-2,
        diag=False,
        solver: LinearSystemSolvers = "cholesky_lu",
        fallback: FallbackLinearSystemSolvers = "safe_diag",
        max_norm: float | None = None,
        validate=False,
        tol: float = 2,
        gd_lr = 1e-2,
        line_search: LineSearches | None = 'brent',
        randomize_every: int = 1,
    ):
        if subspace_ndim == 1: projections = [ProjRandom(1)]
        else:
            projections: list[Any] = [Proj2Masks(subspace_ndim//2)]
            if subspace_ndim % 2 == 1: projections.append(ProjRandom(1))

        modules: list[Any] = [
            Subspace(
                modules = _NewtonFDM(
                    eps = eps,
                    diag = diag,
                    solver=solver,
                    fallback=fallback,
                    validate=validate,
                    tol=tol,
                    gd_lr=gd_lr
                ),
                projections = projections,
                update_every=randomize_every),
        ]
        if max_norm is not None:
            modules.append(ClipNorm(max_norm))

        modules.append(LR(lr))

        if line_search is not None:
            modules.append(get_line_search(line_search))

        super().__init__(params, modules)

