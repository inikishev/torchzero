from typing import Literal
from collections import abc

import torch

from ...utils.derivatives import hessian_list_to_mat, jacobian_and_hessian
from ...tensorlist import TensorList
from ...core import OptimizerModule


def _cholesky_solve(hessian: torch.Tensor, grad: torch.Tensor):
    cholesky, info = torch.linalg.cholesky_ex(hessian) # pylint:disable=not-callable
    if info == 0:
        grad.unsqueeze_(1)
        return torch.cholesky_solve(grad, cholesky), True
    return None, False

def _lu_solve(hessian: torch.Tensor, grad: torch.Tensor):
    try:
        newton_step, info = torch.linalg.solve_ex(hessian, grad) # pylint:disable=not-callable
        if info == 0: return newton_step, True
        return None, False
    except torch.linalg.LinAlgError:
        return None, False


def _cholesky_fallback_lu(hessian: torch.Tensor, grad: torch.Tensor):
    step, success = _cholesky_solve(hessian, grad)
    if not success:
        step, success = _lu_solve(hessian, grad)
    return step, success

def _least_squares_solve(hessian: torch.Tensor, grad: torch.Tensor):
    return torch.linalg.lstsq(hessian, grad)[0], True # pylint:disable=not-callable


def _fallback_gd(hessian:torch.Tensor, grad:torch.Tensor, lr = 1e-2):
    return grad.mul_(1e-2), True

def _fallback_safe_diag(hessian:torch.Tensor, grad:torch.Tensor, lr = 1e-2):
    diag = hessian.diag().reciprocal_().nan_to_num_(1,1,1)
    if torch.all(diag == 1): # fallback to gd
        return _fallback_gd(hessian, grad, lr)
    return grad.mul_(diag * lr), True


def regularize_hessian_(hessian: torch.Tensor, value: float | Literal['eig']):
    """regularize hessian matrix in-place"""
    if value == 'eig':
        value = torch.linalg.eigvalsh(hessian).min().clamp_(max=0).neg_() # pylint:disable=not-callable
    elif value != 0:
        hessian.add_(torch.eye(hessian.shape[0], device=hessian.device,dtype=hessian.dtype), alpha = value)

LinearSystemSolvers = Literal['cholesky', 'lu', 'cholesky_lu', 'lstsq']
FallbackLinearSystemSolvers = Literal['lstsq', 'safe_diag', 'gd']

LINEAR_SYSTEM_SOLVERS = {
    "cholesky": _cholesky_solve,
    "lu": _lu_solve,
    "cholesky_lu": _cholesky_fallback_lu,
    "lstsq": _least_squares_solve,
    "safe_diag": _fallback_safe_diag,
    "gd": _fallback_gd
}

class ExactNewton(OptimizerModule):
    """Peforms an exact Newton step using batched autograd.

    Note that this doesn't support per-group settings.

    Args:
        tikhonov (float, optional):
            tikhonov regularization (constant value added to the diagonal of the hessian).
            Also known as Levenberg-Marquardt regularization. Can be set to 'eig', so it will be set
            to the smallest eigenvalue of the hessian if that value is negative. Defaults to 0.
        solver (Solvers, optional):
            solver for Hx = g. Defaults to "cholesky_lu" (cholesky or LU if it fails).
        fallback (Solvers, optional):
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
        tikhonov: float | Literal['eig'] = 0.0,
        solver: LinearSystemSolvers = "cholesky_lu",
        fallback: FallbackLinearSystemSolvers = "safe_diag",
        validate=False,
        tol: float = 1,
        gd_lr = 1e-2,
        batched_hessian=True,
        diag: bool = False,
    ):
        super().__init__({})
        self.tikhonov: float | Literal['eig'] = tikhonov
        self.batched_hessian = batched_hessian

        self.solver: abc.Callable = LINEAR_SYSTEM_SOLVERS[solver]
        self.fallback: abc.Callable = LINEAR_SYSTEM_SOLVERS[fallback]

        self.validate = validate
        self.gd_lr = gd_lr
        self.tol = tol

        self.diag = diag

    @torch.no_grad
    def step(self, vars):
        if vars.closure is None: raise ValueError("Newton requires a closure to compute the gradient.")

        params = self.get_params()

        # exact hessian via autograd
        with torch.enable_grad():
            vars.fx0 = vars.closure(False)
            grads, hessian = jacobian_and_hessian([vars.fx0], params) # type:ignore
            vars.grad = grads = TensorList(grads).squeeze_(0)
            gvec = grads.to_vec()
            hessian = hessian_list_to_mat(hessian)

        # tikhonov regularization
        regularize_hessian_(hessian, self.tikhonov)

        # calculate newton step
        if self.diag:
            newton_step = gvec / hessian.diag()
        else:
            newton_step, success = self.solver(hessian, gvec)
            if not success:
                newton_step, success = self.fallback(hessian, gvec)
                if not success:
                    newton_step, success = _fallback_gd(hessian, gvec)

        # apply the `_update` method
        vars.ascent = grads.from_vec(newton_step.squeeze_().nan_to_num_(0,0,0))

        # validate if newton step decreased loss
        if self.validate:

            params.sub_(vars.ascent)
            fx1 = vars.closure(False)
            params.add_(vars.ascent)

            # if loss increases, set ascent direction to grad times lr
            if (not fx1.isfinite()) or fx1 - vars.fx0 > vars.fx0 * self.tol: # type:ignore
                vars.ascent = grads.div_(grads.total_vector_norm(2) / self.gd_lr)

        # peform an update with the ascent direction, or pass it to the child.
        return self._update_params_or_step_with_next(vars, params=params)
