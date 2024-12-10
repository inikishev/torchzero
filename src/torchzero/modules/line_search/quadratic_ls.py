import numpy as np
import torch

from ... import tl
from ...core import OptimizationState
from .ls_base import LineSearchBase

_FloatOrTensor = float | torch.Tensor
def _fit_and_minimize_quadratic_2points_grad(x1:_FloatOrTensor,y1:_FloatOrTensor,y1_prime:_FloatOrTensor,x2:_FloatOrTensor,y2:_FloatOrTensor):
    """Fits a quadratic to value and gradient and x1 and value at x2 and returns minima and a parameter."""
    a = (y1_prime * x2 - y2 - y1_prime*x1 + y1) / (x1**2 - x2**2 - 2*x1**2 + 2*x1*x2)
    b = y1_prime - 2*a*x1
    # c = -(a*x1**2 + b*x1 - y1)
    return -b / (2 * a), a

def _ensure_float(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().item()
    elif isinstance(x, np.ndarray): return x.item()
    return float(x)

class MinimizeQuadraticLS(LineSearchBase):
    def __init__(self, lr=1e-2, max_dist = 1e4, validate_step = True, log_lrs = False,):
        super().__init__({"lr": lr}, make_closure=False, maxiter=None, log_lrs=log_lrs)

        self.max_dist = max_dist
        self.validate_step = validate_step

    @torch.no_grad
    def _find_best_lr(self, state: OptimizationState, params: tl.TensorList) -> float:
        if state.closure is None: raise ValueError('QuardaticLS requires closure')
        closure = state.closure
        if state.fx0 is None: state.fx0 = state.closure(False)
        grad = state.grad
        if grad is None: grad = state.ascent_direction # in case we used FDM
        if grad is None: raise ValueError('QuardaticLS requires gradients.')

        params = self.get_params()
        lr: float = self.get_first_group_key('lr') # this doesn't support variable lrs but we still want to support schedulers

        # directional f'(x1)
        y1_prime = grad.total_vector_norm(2)

        # f(x2)
        y2 = self._evaluate_lr_(lr, closure, grad, params)

        # if gradients weren't 0
        if y1_prime != 0:
            xmin, a = _fit_and_minimize_quadratic_2points_grad(
                x1=0,
                y1=state.fx0,
                y1_prime=-y1_prime,
                x2=lr * y1_prime,
                # we stepped in the direction of minus gradient times lr.
                # which is why y1_prime is negative and we multiply x2 by lr.
                y2=y2
            )
            # so we obtained xmin in lr*grad units. We need in lr units.
            xmin = _ensure_float(xmin / y1_prime)

            # make sure curvature is positive
            if a > 0:

                # discard very large steps
                if self.max_dist is None or xmin <= self.max_dist:

                    # if validate_step is enabled, make sure loss didn't increase
                    if self.validate_step:
                        y_val = self._evaluate_lr_(xmin, closure, grad, params)
                        # if it increased, move back to y2.
                        if y_val > y2:
                            return float(lr)

                    return float(xmin)

        return float(lr)

def _fit_and_minimize_quadratic_3points(
    x1: _FloatOrTensor,
    y1: _FloatOrTensor,
    x2: _FloatOrTensor,
    y2: _FloatOrTensor,
    x3: _FloatOrTensor,
    y3: _FloatOrTensor,
):
    """Fits a quadratic to three points."""
    a = (x1*(y3-y2) + x2*(y1-y3) + x3*(y2-y1)) / ((x1-x2) * (x1 - x3) * (x2 - x3))
    b = (y2-y1) / (x2-x1) - a*(x1+x2)
    # c = (y1 - a*x1**2 - b*x1)
    return (-b / (2 * a), a)


def _newton_step_3points(
    xneg: _FloatOrTensor,
    yneg: _FloatOrTensor,
    x0: _FloatOrTensor,
    y0: _FloatOrTensor,
    xpos: _FloatOrTensor, # since points are evenly spaced, xpos is x0 + eps, its turns out unused
    ypos: _FloatOrTensor,
):
    eps = x0 - xneg
    dx = (-yneg + ypos) / (2 * eps)
    ddx = (ypos - 2*y0 + yneg) / (eps**2)

    # xneg is actually x0
    return xneg - dx / ddx, ddx

class MinimizeQuadratic3PointsLS(LineSearchBase):
    def __init__(self, max_dist = 1e4, validate_step = True, log_lrs = False,):
        super().__init__({}, make_closure=False, maxiter=None, log_lrs=log_lrs)

        self.max_dist = max_dist
        self.validate_step = validate_step

    @torch.no_grad
    def _find_best_lr(self, state: OptimizationState, params: tl.TensorList) -> float:
        if state.closure is None: raise ValueError('QuardaticLS requires closure')
        closure = state.closure
        ascent_direction = state.ascent_direction
        if ascent_direction is None: raise ValueError('Ascent direction is None')

        if state.fx0 is None: state.fx0 = state.closure(False)
        params = self.get_params()

        magn = ascent_direction.total_vector_norm(2)

        # make a step in the direction and evaluate f(x2)
        y2 = self._evaluate_lr_(1, closure, ascent_direction, params)

        # make a step in the direction and evaluate f(x3)
        y3 = self._evaluate_lr_(2, closure, ascent_direction, params)

        # if gradients weren't 0
        xmin, a = _newton_step_3points(
            0, state.fx0,
            # we stepped in the direction of minus ascent_direction.
            magn, y2,
            magn * 2, y3
        )
        xmin = _ensure_float(xmin / magn)

        # make sure curvature is positive
        if a > 0:

            # discard very large steps
            if self.max_dist is None or xmin <= self.max_dist:

                # if validate_step is enabled, make sure loss didn't increase
                if self.validate_step:
                    y_val = self._evaluate_lr_(xmin, closure, ascent_direction, params)
                    # if it increased, move back to y2.
                    if y_val > y2 or y_val > y3:
                        if y3 > y2: return 1
                        else: return 2

                return xmin

        if y3 > y2: return 1
        else: return 2
