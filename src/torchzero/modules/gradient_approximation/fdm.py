import typing as T

import torch

from ...python_tools import ScalarType
from ...tensorlist import TensorList
from ...core import ClosureType, OptimizerModule, OptimizationState
from ._fd_formulas import _FD_Formulas


def _two_point_fd_(closure: ClosureType, idx: int, pvec: torch.Tensor, gvec: torch.Tensor, eps: ScalarType, fx0: ScalarType, ):
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

def _two_point_bd_(closure: ClosureType, idx: int, pvec: torch.Tensor, gvec: torch.Tensor, eps: ScalarType, fx0: ScalarType, ):
    pvec[idx] += eps
    fx1 = closure(False)
    gvec[idx] = (fx0 - fx1) / eps
    pvec[idx] -= eps
    return fx0

def _two_point_cd_(closure: ClosureType, idx: int, pvec: torch.Tensor, gvec: torch.Tensor, eps: ScalarType, fx0 = None, ):
    pvec[idx] += eps
    fxplus = closure(False)
    pvec[idx] -= eps * 2
    fxminus = closure(False)
    gvec[idx] = (fxplus - fxminus) / (2 * eps)
    pvec[idx] += eps
    return fxplus

def _three_point_fd_(closure: ClosureType, idx: int, pvec: torch.Tensor, gvec: torch.Tensor, eps: ScalarType, fx0: ScalarType, ):
    pvec[idx] += eps
    fx1 = closure(False)
    pvec[idx] += eps
    fx2 = closure(False)
    gvec[idx] = (-3*fx0 + 4*fx1 - fx2) / (2 * eps)
    pvec[idx] -= 2 * eps
    return fx0

def _three_point_bd_(closure: ClosureType, idx: int, pvec: torch.Tensor, gvec: torch.Tensor, eps: ScalarType, fx0: ScalarType, ):
    pvec[idx] -= eps
    fx1 = closure(False)
    pvec[idx] -= eps
    fx2 = closure(False)
    gvec[idx] = (fx2 - 4*fx1 + 3*fx0) / (2 * eps)
    pvec[idx] += 2 * eps
    return fx0



class FDM(OptimizerModule):
    def __init__(
        self,
        eps: float = 1e-5,
        formula: _FD_Formulas = "forward",
        n_points: T.Literal[2, 3] = 2,
        make_closure=False,
    ):
        """Gradient approximation via finite difference.

        Args:
            eps (float, optional): finite difference epsilon. Defaults to 1e-5.
            formula (_FD_Formulas, optional): finite difference formula. Defaults to 'forward'.
            n_points (T.Literal[2, 3], optional): number of points, 2 or 3. Defaults to 2.
            make_closure (bool, optional): if True, this makes a new closure that sets .grad attribute on each call
                with `backward = True`. If False, this simply returns the estimated gradients as the ascent direction.
                Note that with `True` this will perform 1 additional evaluation per step with the `central` formula.
        """
        defaults = dict(eps = eps)
        super().__init__(defaults)

        self.make_closure = make_closure

        if formula == 'central':
            self._finite_difference_ = _two_point_cd_ # this is both 2 and 3 point formula
            self._requires_fx0 = False

        elif formula == 'forward':
            if n_points == 2: self._finite_difference_ = _two_point_fd_
            else: self._finite_difference_ = _three_point_fd_
            self._requires_fx0 = True

        elif formula == 'backward':
            if n_points == 2: self._finite_difference_ = _two_point_bd_
            else: self._finite_difference_ = _three_point_bd_
            self._requires_fx0 = True

        else: raise ValueError(f'{formula} is not valid.')

    @torch.no_grad
    def _make_closure_step(self, state: OptimizationState, params: TensorList, epsilons: TensorList):
        """Makes a new closure that sets .grad attribute on backward=True."""
        closure = state.closure
        if closure is None: raise ValueError('FDM requires closure.')

        # the new closure sets .grad attribute to finite difference-approximated gradients
        @torch.no_grad
        def fdm_closure(backward = True):
            # closure must always evaluate the loss
            # regardless of whether we need it at fx0 or not
            loss = closure(False)

            if backward:
                grads = params.zeros_like()
                # evaluate gradients via finite differences.
                for p, g, eps in zip(params, grads, epsilons):
                    flat_param = p.view(-1)
                    flat_grad = g.view(-1)
                    for idx in range(flat_param.numel()):
                        self._finite_difference_(closure, idx, flat_param, flat_grad, eps, loss, ) # type:ignore

                    # set the grad attribute
                    # (accumulation doesn't make sense here as closure always calls zero_grad)
                    p.grad = g.view_as(p)

            return loss

        # FDM always passes the approximated gradients to its child.
        if self.next_module is None: raise ValueError("FDM with `make_closure=True` requires a child.")
        state.closure = fdm_closure
        return self.next_module.step(state)


    @torch.no_grad
    def _make_ascent_direction_step(self, state: OptimizationState, params: TensorList, epsilons: TensorList):
        """Returns a new ascent direction."""
        # evaluate fx0 if it is needed for forward and backward differences.
        closure = state.closure
        if closure is None: raise ValueError('FDM requires closure.')
        if state.fx0 is None and self._requires_fx0: state.fx0 = closure(False)

        # evaluate gradients via finite differences.
        grads = params.zeros_like()
        for p, g, eps in zip(params, grads, epsilons):
            flat_param = p.view(-1)
            flat_grad = g.view(-1)
            for idx in range(flat_param.numel()):
                state.fx0_approx = self._finite_difference_(closure, idx, flat_param, flat_grad, eps, state.fx0, ) # type:ignore

        # update params or pass the gradients to the child.
        state.ascent = grads
        return self._update_params_or_step_with_next(state, params)

    def step(self, state):

        params = self.get_params()
        epsilons = self.get_group_key('eps')

        if self.make_closure:
            return self._make_closure_step(state, params = params, epsilons = epsilons)
        else:
            if state.ascent is not None: raise ValueError('FDM with `make_closure=False` does not accept ascent direction.')
            return self._make_ascent_direction_step(state, params = params, epsilons = epsilons)