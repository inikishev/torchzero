import typing as T

import torch

from ...python_tools import ScalarType
from ...tensorlist import TensorList
from ...core import ClosureType, OptimizerModule
from ..second_order.newton import (LINEAR_SYSTEM_SOLVERS,
                                   FallbackLinearSystemSolvers,
                                   LinearSystemSolvers, _fallback_gd)
from ._fd_formulas import _FD_Formulas


def _three_point_2cd_(
    closure: ClosureType,
    idx1: int,
    idx2: int,
    p1: torch.Tensor,
    p2: torch.Tensor,
    g1: torch.Tensor,
    hessian: torch.Tensor,
    eps1: ScalarType,
    eps2: ScalarType,
    i1: int,
    i2: int,
    fx0: ScalarType,
):
    """Second order three point finite differences (same signature for all other 2nd order finite differences functions).

    Args:
        closure (ClosureType): _description_
        idx1 (int): _description_
        idx2 (int): _description_
        p1 (torch.Tensor): _description_
        p2 (torch.Tensor): _description_
        g1 (torch.Tensor): _description_
        g2 (torch.Tensor): _description_
        hessian (torch.Tensor): _description_
        eps1 (ScalarType): _description_
        eps2 (ScalarType): _description_
        i1 (int): _description_
        i23 (int): _description_
        fx0 (ScalarType): _description_

    """
    # same param
    if i1 == i2 and idx1 == idx2:
        p1[idx1] += eps1
        fxplus = closure(False)

        p1[idx1] -= 2*eps1
        fxminus = closure(False)

        p1[idx1] += eps1

        g1[idx1] = (fxplus - fxminus) / (2 * eps1)
        hessian[i1, i2] = (fxplus - 2*fx0 + fxminus) / eps1**2

    else:
        p1[idx1] += eps1
        p2[idx2] += eps2
        fxpp = closure(False)
        p1[idx1] -= eps1*2
        fxnp = closure(False)
        p2[idx2] -= eps2*2
        fxnn = closure(False)
        p1[idx1] += eps1*2
        fxpn = closure(False)

        p1[idx1] -= eps1
        p2[idx2] += eps2

        hessian[i1, i2] = (fxpp - fxpn - fxnp + fxnn) / (4 * eps1 * eps2)


class NewtonFDM(OptimizerModule):
    def __init__(
        self,
        eps: float = 1e-2,
        diag=False,
        solver: LinearSystemSolvers = "cholesky_lu",
        fallback: FallbackLinearSystemSolvers = "safe_diag",
        validate=False,
        tol: float = 1,
        gd_lr = 1e-2,
    ):
        """Newton method with gradient and hessian approximated via finite difference.

        Args:
            eps (float, optional): epsilon for finite difference.
                Note that with float32 this needs to be quite high to avoid numerical instability. Defaults to 1e-2.
            diag (bool, optional): whether to only approximate diagonal elements of the hessian.
                If true, ignores `solver` and `fallback`. Defaults to False.
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

        """
        defaults = dict(eps = eps)
        super().__init__(defaults)
        self.diag = diag
        self.solver = LINEAR_SYSTEM_SOLVERS[solver]
        self.fallback = LINEAR_SYSTEM_SOLVERS[fallback]

        self.validate = validate
        self.gd_lr = gd_lr
        self.tol = tol

    @torch.no_grad
    def step(self, state):
        """Returns a new ascent direction."""
        if state.closure is None: raise ValueError('NewtonFDM requires a closure.')
        if state.ascent_direction is not None: raise ValueError('NewtonFDM got ascent direction')

        params = self.get_params()
        epsilons = self.get_group_key('eps')

        # evaluate fx0.
        if state.fx0 is None: state.fx0 = state.closure(False)

        # evaluate gradients and hessian via finite differences.
        grads = params.zeros_like()
        numel = params.total_numel()
        hessian = torch.zeros((numel, numel), dtype = params[0].dtype, device = params[0].device)

        cur1 = 0
        for p1, g1, eps1 in zip(params, grads, epsilons):
            flat_param1 = p1.view(-1)
            flat_grad1 = g1.view(-1)
            for idx1 in range(flat_param1.numel()):

                cur2 = 0
                for p2, eps2 in zip(params, epsilons):

                    flat_param2 = p2.view(-1)
                    for idx2 in range(flat_param2.numel()):
                        if self.diag and (idx1 != idx2 or cur1 != cur2):
                            cur2 += 1
                            continue
                        _three_point_2cd_(
                            closure = state.closure,
                            idx1 = idx1,
                            idx2 = idx2,
                            p1 = flat_param1,
                            p2 = flat_param2,
                            g1 = flat_grad1,
                            hessian = hessian,
                            eps1 = eps1,
                            eps2 = eps2,
                            fx0 = state.fx0,
                            i1 = cur1,
                            i2 = cur2,
                        )
                        cur2 += 1
                cur1 += 1

        gvec = grads.to_vec()
        if self.diag:
            hdiag = hessian.diag()
            hdiag[hdiag == 0] = 1
            newton_step = gvec / hdiag
        else:
            newton_step, success = self.solver(hessian, gvec)
            if not success:
                newton_step, success = self.fallback(hessian, gvec)
                if not success:
                    newton_step, success = _fallback_gd(hessian, gvec)

        # update params or pass the gradients to the child.
        state.ascent_direction = grads.from_vec(newton_step)


        # validate if newton step decreased loss
        if self.validate:

            params.sub_(state.ascent_direction)
            fx1 = state.closure(False)
            params.add_(state.ascent_direction)

            # if loss increases, set ascent direction to gvec times lr
            if fx1 - state.fx0 > state.fx0 * self.tol:
                state.ascent_direction = grads.from_vec(gvec) * self.gd_lr

        return self._update_params_or_step_with_child(state, params)
