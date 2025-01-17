from typing import Literal, Any
from warnings import warn
import torch

from ...utils.python_tools import _ScalarLoss
from ...tensorlist import TensorList
from ...core import _ClosureType, OptimizerModule, OptimizationState
from ._fd_formulas import _FD_Formulas
from .base_approximator import GradientApproximatorBase

def _two_point_fd_(closure: _ClosureType, idx: int, pvec: torch.Tensor, gvec: torch.Tensor, eps: _ScalarLoss, fx0: _ScalarLoss, ):
    """Two point finite difference (same signature for all other finite differences functions).

    Args:
        closure (Callable): A closure that reevaluates the model and returns the loss.
        idx (int): Flat index of the current parameter.
        pvec (Tensor): Flattened view of the current parameter tensor.
        gvec (Tensor): Flattened view of the current parameter tensor gradient.
        eps (float): Finite difference epsilon.
        fx0 (ScalarType): Loss at fx0, to avoid reevaluating it each time. On some functions can be None when it isn't needed.

    Returns:
        This modifies `gvec` in place.
        This returns loss, not necessarily at fx0 (for example central difference never evaluate at fx0).
        So this should be assigned to fx0_approx.
    """
    pvec[idx] += eps
    fx1 = closure(False)
    gvec[idx] = (fx1 - fx0) / eps
    pvec[idx] -= eps
    return fx0

def _two_point_bd_(closure: _ClosureType, idx: int, pvec: torch.Tensor, gvec: torch.Tensor, eps: _ScalarLoss, fx0: _ScalarLoss, ):
    pvec[idx] += eps
    fx1 = closure(False)
    gvec[idx] = (fx0 - fx1) / eps
    pvec[idx] -= eps
    return fx0

def _two_point_cd_(closure: _ClosureType, idx: int, pvec: torch.Tensor, gvec: torch.Tensor, eps: _ScalarLoss, fx0 = None, ):
    pvec[idx] += eps
    fxplus = closure(False)
    pvec[idx] -= eps * 2
    fxminus = closure(False)
    gvec[idx] = (fxplus - fxminus) / (2 * eps)
    pvec[idx] += eps
    return fxplus

def _three_point_fd_(closure: _ClosureType, idx: int, pvec: torch.Tensor, gvec: torch.Tensor, eps: _ScalarLoss, fx0: _ScalarLoss, ):
    pvec[idx] += eps
    fx1 = closure(False)
    pvec[idx] += eps
    fx2 = closure(False)
    gvec[idx] = (-3*fx0 + 4*fx1 - fx2) / (2 * eps)
    pvec[idx] -= 2 * eps
    return fx0

def _three_point_bd_(closure: _ClosureType, idx: int, pvec: torch.Tensor, gvec: torch.Tensor, eps: _ScalarLoss, fx0: _ScalarLoss, ):
    pvec[idx] -= eps
    fx1 = closure(False)
    pvec[idx] -= eps
    fx2 = closure(False)
    gvec[idx] = (fx2 - 4*fx1 + 3*fx0) / (2 * eps)
    pvec[idx] += 2 * eps
    return fx0


class FDM(GradientApproximatorBase):
    """Gradient approximation via finite difference.

    This performs :math:`num_parameters + 1` or :math:`num_parameters * 2` evaluations per step, depending on formula.

    Args:
        eps (float, optional): finite difference epsilon. Defaults to 1e-5.
        formula (_FD_Formulas, optional): finite difference formula. Defaults to 'forward'.
        n_points (T.Literal[2, 3], optional): number of points, 2 or 3. Defaults to 2.
        target (str, optional):
            determines what this module sets.

            "ascent" - it creates a new ascent direction but doesn't treat is as gradient.

            "grad" - it creates the gradient and sets it to `.grad` attributes (default).

            "closure" - it makes a new closure that sets the estimated gradient to the `.grad` attributes.
    """
    def __init__(
        self,
        eps: float = 1e-5,
        formula: _FD_Formulas = "forward",
        n_points: Literal[2, 3] = 2,
        target: Literal["ascent", "grad", "closure"] = "grad",
    ):
        defaults = dict(eps = eps)

        if formula == 'central':
            self._finite_difference_ = _two_point_cd_ # this is both 2 and 3 point formula
            requires_fx0 = False

        elif formula == 'forward':
            if n_points == 2: self._finite_difference_ = _two_point_fd_
            else: self._finite_difference_ = _three_point_fd_
            requires_fx0 = True

        elif formula == 'backward':
            if n_points == 2: self._finite_difference_ = _two_point_bd_
            else: self._finite_difference_ = _three_point_bd_
            requires_fx0 = True

        else: raise ValueError(f'{formula} is not valid.')

        super().__init__(defaults, requires_fx0=requires_fx0, target = target)

    @torch.no_grad
    def _make_ascent(self, closure, params, fx0):
        grads = params.zeros_like()
        epsilons = self.get_group_key('eps')

        fx0_approx = None
        for p, g, eps in zip(params, grads, epsilons):
            flat_param = p.view(-1)
            flat_grad = g.view(-1)
            for idx in range(flat_param.numel()):
                fx0_approx = self._finite_difference_(closure, idx, flat_param, flat_grad, eps, fx0)

        return grads, fx0, fx0_approx
