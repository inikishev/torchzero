import warnings

import torch

from ...core import Chainable, Transform
from ...linalg import linear_operator
from ...utils import vec_to_tensors
from ...utils.derivatives import flatten_jacobian, jacobian_wrt


class SumOfSquares(Transform):
    """Sets loss to be the sum of squares of values returned by the closure.

    This is meant to be used to test least squares methods against ordinary minimization methods.

    To use this, the closure should return a vector of values to minimize sum of squares of.
    Please add the ``backward`` argument, it will always be False but it is required.
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad
    def update_states(self, objective, states, settings):
        closure = objective.closure

        if closure is not None:

            def sos_closure(backward=True):
                if backward:
                    objective.zero_grad()
                    with torch.enable_grad():
                        loss = closure(False)
                        loss = loss.pow(2).sum()
                        loss.backward()
                    return loss

                loss = closure(False)
                return loss.pow(2).sum()

            objective.closure = sos_closure

        if objective.loss is not None:
            objective.loss = objective.loss.pow(2).sum()

        if objective.loss_approx is not None:
            objective.loss_approx = objective.loss_approx.pow(2).sum()

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        return objective

class GaussNewton(Transform):
    """Gauss-newton method.

    To use this, the closure should return a vector of values to minimize sum of squares of.
    Please add the ``backward`` argument, it will always be False but it is required.
    Gradients will be calculated via batched autograd within this module, you don't need to
    implement the backward pass. Please see below for an example.

    Note:
        This method requires ``ndim^2`` memory, however, if it is used within ``tz.m.TrustCG`` trust region,
        the memory requirement is ``ndim*m``, where ``m`` is number of values in the output.

    Args:
        reg (float, optional): regularization parameter. Defaults to 1e-8.
        update_freq (int, optional):
            frequency of computing the jacobian. When jacobian is not computed, only residuals are computed and updated.
            Defaults to 1.
        batched (bool, optional): whether to use vmapping. Defaults to True.

    Examples:

    minimizing the rosenbrock function:
    ```python
    def rosenbrock(X):
        x1, x2 = X
        return torch.stack([(1 - x1), 100 * (x2 - x1**2)])

    X = torch.tensor([-1.1, 2.5], requires_grad=True)
    opt = tz.Optimizer([X], tz.m.GaussNewton(), tz.m.Backtracking())

    # define the closure for line search
    def closure(backward=True):
        return rosenbrock(X)

    # minimize
    for iter in range(10):
        loss = opt.step(closure)
        print(f'{loss = }')
    ```

    training a neural network with a matrix-free GN trust region:
    ```python
    X = torch.randn(64, 20)
    y = torch.randn(64, 10)

    model = nn.Sequential(nn.Linear(20, 64), nn.ELU(), nn.Linear(64, 10))
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.TrustCG(tz.m.GaussNewton()),
    )

    def closure(backward=True):
        y_hat = model(X) # (64, 10)
        return (y_hat - y).pow(2).mean(0) # (10, )

    for i in range(100):
        losses = opt.step(closure)
        if i % 10 == 0:
            print(f'{losses.mean() = }')
    ```
    """
    def __init__(self, reg:float = 1e-8, update_freq: int= 1, batched:bool=True, inner: Chainable | None = None):
        defaults=dict(update_freq=update_freq,batched=batched, reg=reg)
        super().__init__(defaults=defaults)
        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        fs = settings[0]
        params = objective.params
        closure = objective.closure
        batched = fs['batched']
        update_freq = fs['update_freq']

        # compute residuals
        r = objective.loss
        if r is None:
            assert closure is not None
            with torch.enable_grad():
                r = objective.get_loss(backward=False) # n_residuals
                assert isinstance(r, torch.Tensor)

        if r.numel() == 1:
            r = r.view(1,1)
            warnings.warn("Gauss-newton got a single residual. Make sure objective function returns a vector of residuals.")

        # set sum of squares scalar loss and it's gradient to objective
        objective.loss = r.pow(2).sum()

        step = self.increment_counter("step", start=0)

        if step % update_freq == 0:

            # compute jacobian
            with torch.enable_grad():
                J_list = jacobian_wrt([r.ravel()], params, batched=batched)

            J = self.global_state["J"] = flatten_jacobian(J_list) # (n_residuals, ndim)

        else:
            J = self.global_state["J"]

        Jr = J.T @ r.detach() # (ndim)

        # if there are more residuals, solve (J^T J)x = J^T r, so we need Jr
        # otherwise solve (J J^T)z = r and set x = J^T z, so we need r
        n_residuals, ndim = J.shape
        if n_residuals >= ndim or "inner" in self.children:
            self.global_state["Jr"] = Jr

        else:
            self.global_state["r"] = r

        objective.grads = vec_to_tensors(Jr, objective.params)

        # set closure to calculate sum of squares for line searches etc
        if closure is not None:
            def sos_closure(backward=True):

                if backward:
                    objective.zero_grad()
                    with torch.enable_grad():
                        loss = closure(False).pow(2).sum()
                        loss.backward()
                    return loss

                loss = closure(False).pow(2).sum()
                return loss

            objective.closure = sos_closure

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        fs = settings[0]
        reg = fs['reg']

        J: torch.Tensor = self.global_state['J']
        nresiduals, ndim = J.shape
        if nresiduals >= ndim or "inner" in self.children:

            # (J^T J)v = J^T r
            Jr: torch.Tensor = self.global_state['Jr']

            # inner step
            if "inner" in self.children:

                # var.grad is set to unflattened Jr
                assert objective.grads is not None
                objective = self.inner_step("inner", objective, must_exist=True)
                Jr_list = objective.get_updates()
                Jr = torch.cat([t.ravel() for t in Jr_list])

            JtJ = J.T @ J # (ndim, ndim)
            if reg != 0:
                JtJ.add_(torch.eye(JtJ.size(0), device=JtJ.device, dtype=JtJ.dtype).mul_(reg))

            if nresiduals >= ndim:
                v, info = torch.linalg.solve_ex(JtJ, Jr) # pylint:disable=not-callable
            else:
                v = torch.linalg.lstsq(JtJ, Jr).solution # pylint:disable=not-callable

            objective.updates = vec_to_tensors(v, objective.params)
            return objective

        # else:
        # solve (J J^T)z = r and set v = J^T z
        # we need (J^T J)v = J^T r
        # if z is solution to (G G^T)z = r, and v = J^T z
        # then (J^T J)v = (J^T J) (J^T z) = J^T (J J^T) z = J^T r
        # therefore (J^T J)v = J^T r
        # also this gives a minimum norm solution

        r = self.global_state['r']

        JJT = J @ J.T # (nresiduals, nresiduals)
        if reg != 0:
            JJT.add_(torch.eye(JJT.size(0), device=JJT.device, dtype=JJT.dtype).mul_(reg))

        z, info = torch.linalg.solve_ex(JJT, r) # pylint:disable=not-callable
        v = J.T @ z

        objective.updates = vec_to_tensors(v, objective.params)
        return objective

    def get_H(self, objective=...):
        J = self.global_state['J']
        return linear_operator.AtA(J)
