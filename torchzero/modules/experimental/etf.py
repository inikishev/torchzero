from typing import cast
import warnings

import torch

from ...core import Module
from ...utils import vec_to_tensors, vec_to_tensors_, as_tensorlist


class ExponentialTrajectoryFit(Module):
    """A method.

    .. warning::
        Experimental.
    """
    def __init__(self, step_size=1e-2, adaptive:bool=True):
        defaults = dict(step_size = step_size,adaptive=adaptive)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        closure = var.closure
        assert closure is not None
        step_size = self.settings[var.params[0]]['step_size']
        adaptive = self.settings[var.params[0]]['adaptive']


        # 1. perform 3 GD steps to obtain 4 points
        points = [torch.cat([p.view(-1) for p in var.params])]
        for i in range(3):
            if i == 0:
                grad = var.get_grad()
                if adaptive:
                    step_size /= as_tensorlist(grad).abs().global_mean().clip(min=1e-4)

            else:
                with torch.enable_grad(): closure()
                grad = [cast(torch.Tensor, p.grad) for p in var.params]

            # GD step
            torch._foreach_sub_(var.params, grad, alpha=step_size)

            points.append(torch.cat([p.view(-1) for p in var.params]))

        assert len(points) == 4, len(points)
        x0, x1, x2, x3 = points
        dim = x0.numel()

        # 2. fit a generalized exponential curve
        d0 = (x1 - x0).unsqueeze(1) # column vectors
        d1 = (x2 - x1).unsqueeze(1)
        d2 = (x3 - x2).unsqueeze(1)

        # cat
        D1 = torch.cat([d0, d1], dim=1)
        D2 = torch.cat([d1, d2], dim=1)

        # if points are collinear this will happen on sphere and a quadratic "line search" will minimize it
        if x0.numel() >= 2:
            if torch.linalg.matrix_rank(D1) < 2: # pylint:disable=not-callable
                pass # need to put a quadratic fit there

        M = D2 @ torch.linalg.pinv(D1) # pylint:disable=not-callable # this defines the curve

        # now we can predict x*
        I = torch.eye(dim, device=x0.device, dtype=x0.dtype)
        B = I - M
        z = x1 - M @ x0

        x_star = torch.linalg.lstsq(B, z).solution # pylint:disable=not-callable

        vec_to_tensors_(x0, var.params)
        difference = torch._foreach_sub(var.params, vec_to_tensors(x_star, var.params))
        var.update = list(difference)
        return var



class ExponentialTrajectoryFitV2(Module):
    """Should be better than one above, except it isn't.

    .. warning::
        Experimental.

    """
    def __init__(self, step_size=1e-3, num_steps: int= 4, adaptive:bool=True):
        defaults = dict(step_size = step_size, num_steps=num_steps, adaptive=adaptive)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        closure = var.closure
        assert closure is not None
        step_size = self.settings[var.params[0]]['step_size']
        num_steps = self.settings[var.params[0]]['num_steps']
        adaptive = self.settings[var.params[0]]['adaptive']

        # 1. perform 3 GD steps to obtain 4 points (or more)
        grad = var.get_grad()
        if adaptive:
            step_size /= as_tensorlist(grad).abs().global_mean().clip(min=1e-4)

        points = [torch.cat([p.view(-1) for p in var.params])]
        point_grads = [torch.cat([g.view(-1) for g in grad])]

        for i in range(num_steps):
            # GD step
            torch._foreach_sub_(var.params, grad, alpha=step_size)

            points.append(torch.cat([p.view(-1) for p in var.params]))

            closure(backward=True)
            grad = [cast(torch.Tensor, p.grad) for p in var.params]
            point_grads.append(torch.cat([g.view(-1) for g in grad]))


        X = torch.stack(points, 1) # dim, num_steps+1
        G = torch.stack(point_grads, 1)
        dim = points[0].numel()

        X = torch.cat([X, torch.ones(1, num_steps+1, dtype=G.dtype, device=G.device)])

        P = G @ torch.linalg.pinv(X) # pylint:disable=not-callable
        A = P[:, :dim]
        b = -P[:, dim]

        # symmetrize
        A = 0.5 * (A + A.T)

        # predict x*
        x_star = torch.linalg.lstsq(A, b).solution # pylint:disable=not-callable

        vec_to_tensors_(points[0], var.params)
        difference = torch._foreach_sub(var.params, vec_to_tensors(x_star, var.params))
        var.update = list(difference)
        return var




def _fit_exponential(y0, y1, y2):
    """x0, x1 and x2 are assumed to be 0, 1, 2"""
    r = (y2 - y1) / (y1 - y0)
    ones = r==1
    r[ones] = 0
    B = (y1 - y0) / (r - 1)
    A = y0 - B

    A[ones] = 0
    B[ones] = 0
    return A, B, r

class PointwiseExponential(Module):
    """A stupid method (for my youtube channel).

    .. warning::
        Experimental.
    """
    def __init__(self, step_size: float = 1e-3, reg: float = 1e-2, steps = 10000):
        defaults = dict(reg=reg, steps=steps, step_size=step_size)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        closure = var.closure
        assert closure is not None
        settings = self.settings[var.params[0]]
        step_size = settings['step_size']
        reg = settings['reg']
        steps = settings['steps']

        # 1. perform 2 GD steps to obtain 3 points
        points = [torch.cat([p.view(-1) for p in var.params])]
        for i in range(2):
            if i == 0: grad = var.get_grad()
            else:
                with torch.enable_grad(): closure()
                grad = [cast(torch.Tensor, p.grad) for p in var.params]

            # GD step
            torch._foreach_sub_(var.params, grad, alpha=step_size)

            points.append(torch.cat([p.view(-1) for p in var.params]))

        assert len(points) == 3, len(points)
        y0, y1, y2 = points

        A, B, r = _fit_exponential(y0, y1, y2)
        r = r.clip(max = 1-reg)
        x_star = A + B * r**steps

        vec_to_tensors_(y0, var.params)
        difference = torch._foreach_sub(var.params, vec_to_tensors(x_star, var.params))
        var.update = list(difference)
        return var
