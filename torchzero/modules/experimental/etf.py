from typing import cast
import warnings

import torch

from ...core import Module
from ...utils import vec_to_tensors, vec_to_tensors_


class ExponentialTrajectoryFit(Module):
    """I think this is Anderson Acceleration? But idk"""
    def __init__(self, step_size=1e-3):
        defaults = dict(step_size = step_size)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, vars):
        closure = vars.closure
        assert closure is not None
        step_size = self.settings[vars.params[0]]['step_size']

        # 1. perform 3 GD steps to obtain 4 points
        points = [torch.cat([p.view(-1) for p in vars.params])]
        for i in range(3):
            if i == 0: grad = vars.get_grad()
            else:
                closure()
                grad = [cast(torch.Tensor, p.grad) for p in vars.params]

            # GD step
            torch._foreach_sub_(vars.params, grad, alpha=step_size)

            points.append(torch.cat([p.view(-1) for p in vars.params]))

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

        vec_to_tensors_(x0, vars.params)
        difference = torch._foreach_sub(vars.params, vec_to_tensors(x_star, vars.params))
        vars.update = list(difference)
        return vars



class ExponentialTrajectoryFitV2(Module):
    """Should be better but isn't"""
    def __init__(self, step_size=1e-3, num_steps: int= 4):
        defaults = dict(step_size = step_size, num_steps=num_steps)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, vars):
        closure = vars.closure
        assert closure is not None
        step_size = self.settings[vars.params[0]]['step_size']
        num_steps = self.settings[vars.params[0]]['num_steps']

        # 1. perform 3 GD steps to obtain 4 points (or more)
        grad = vars.get_grad()
        points = [torch.cat([p.view(-1) for p in vars.params])]
        point_grads = [torch.cat([g.view(-1) for g in grad])]

        for i in range(num_steps):
            # GD step
            torch._foreach_sub_(vars.params, grad, alpha=step_size)

            points.append(torch.cat([p.view(-1) for p in vars.params]))

            closure(backward=True)
            grad = [cast(torch.Tensor, p.grad) for p in vars.params]
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

        vec_to_tensors_(points[0], vars.params)
        difference = torch._foreach_sub(vars.params, vec_to_tensors(x_star, vars.params))
        vars.update = list(difference)
        return vars