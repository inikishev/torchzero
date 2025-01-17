import bisect

import numpy as np
import torch

from ... import tl
from ...core import OptimizationState
from ..line_search.base_ls import LineSearchBase

_FloatOrTensor = float | torch.Tensor

def _ensure_float(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().item()
    elif isinstance(x, np.ndarray): return x.item()
    return float(x)

class Point:
    def __init__(self, x, fx, dfx = None):
        self.x = x
        self.fx = fx
        self.dfx = dfx

    def __repr__(self):
        return f'Point(x={self.x:.2f}, fx={self.fx:.2f})'

def _step_2poins(x1, f1, df1, x2, f2):
    # we have two points and one derivative
    # minimize the quadratic to obtain 3rd point and perform bracketing
    a = (df1 * x2 - f2 - df1*x1 + f1) / (x1**2 - x2**2 - 2*x1**2 + 2*x1*x2)
    b = df1 - 2*a*x1
    # c = -(a*x1**2 + b*x1 - y1)
    return -b / (2 * a), a

class QuadraticInterpolation2Point(LineSearchBase):
    """This is WIP, please don't use yet!
    Use `torchzero.modules.MinimizeQuadraticLS` and `torchzero.modules.MinimizeQuadratic3PointsLS` instead.

    Args:
        lr (_type_, optional): _description_. Defaults to 1e-2.
        log_lrs (bool, optional): _description_. Defaults to False.
        max_evals (int, optional): _description_. Defaults to 2.
        min_dist (_type_, optional): _description_. Defaults to 1e-2.
    """
    def __init__(self, lr=1e-2, log_lrs = False, max_evals = 2, min_dist = 1e-2,):
        super().__init__({"lr": lr}, maxiter=None, log_lrs=log_lrs)
        self.max_evals = max_evals
        self.min_dist = min_dist

    @torch.no_grad
    def _find_best_lr(self, state: OptimizationState, params: tl.TensorList) -> float:
        if state.closure is None: raise ValueError('QuardaticLS requires closure')
        closure = state.closure
        if state.fx0 is None: state.fx0 = state.closure(False)
        grad = state.grad
        if grad is None: grad = state.ascent # in case we used FDM
        if grad is None: raise ValueError('QuardaticLS requires gradients.')

        params = self.get_params()
        lr: float = self.get_first_group_key('lr') # this doesn't support variable lrs but we still want to support schedulers

        # directional f'(x0)
        # for each lr we step by this much
        dfx0 = magn = grad.total_vector_norm(2)

        # f(x1)
        fx1 = self._evaluate_lr_(lr, closure, grad, params)

        # make min_dist relative
        min_dist = abs(lr) * self.min_dist
        points = sorted([Point(0, _ensure_float(state.fx0), dfx0), Point(lr, _ensure_float(fx1))], key = lambda x: x.fx)

        for i in range(self.max_evals):
            # find new point
            p1, p2 = points
            if p1.dfx is None: p1, p2 = p2, p1
            xmin, curvature = _step_2poins(p1.x * magn, p1.fx, -p1.dfx, p2.x * magn, p2.fx) # type:ignore
            xmin = _ensure_float(xmin/magn)
            print(f'{xmin = }', f'{curvature = }, n_evals = {i+1}')

            # if max_evals = 1, we just minimize a quadratic once
            if i == self.max_evals - 1:
                if curvature > 0: return xmin
                return lr

            # TODO: handle negative curvature
            # if curvature < 0:
            #     if points[0].x == 0: return lr
            #     return points[0].x

            # evaluate value and gradients at new point
            fxmin = self._evaluate_lr_(xmin, closure, grad, params, backward=True)
            dfxmin = -(params.grad * grad).total_sum()

            # insort new point
            bisect.insort(points, Point(xmin, _ensure_float(fxmin), dfxmin), key = lambda x: x.fx)

            # pick 2 best points to find the new bracketing interval
            points = sorted(points, key = lambda x: x.fx)[:2]
            # TODO: new point might be worse than 2 existing ones which would lead to stagnation

            # if points are too close, end the loop
            if abs(points[0].x - points[1].x) < min_dist: break

        return points[0].x