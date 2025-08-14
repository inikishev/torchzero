import torch
from ...core import Module

from ...utils.derivatives import jacobian_wrt, flatten_jacobian
from ...utils import vec_to_tensors
from ...utils.linalg import linear_operator
class SumOfSquares(Module):
    """Sets loss to be the sum of squares of values returned by the closure.

    This is meant to be used to test least squares methods against ordinary minimization methods.

    To use this, the closure should return a vector of values to minimize sum of squares of.
    Please add the `backward` argument, it will always be False but it is required.
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad
    def step(self, var):
        closure = var.closure

        if closure is not None:
            def sos_closure(backward=True):
                if backward:
                    var.zero_grad()
                    with torch.enable_grad():
                        loss = closure(False)
                        loss = loss.pow(2).sum()
                        loss.backward()
                    return loss

                loss = closure(False)
                return loss.pow(2).sum()

            var.closure = sos_closure

        if var.loss is not None:
            var.loss = var.loss.pow(2).sum()

        if var.loss_approx is not None:
            var.loss_approx = var.loss_approx.pow(2).sum()

        return var


class GaussNewton(Module):
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
        batched (bool, optional): whether to use vmapping. Defaults to True.

    Examples:

    minimizing the rosenbrock function:
    ```python
    def rosenbrock(X):
        x1, x2 = X
        return torch.stack([(1 - x1), 100 * (x2 - x1**2)])

    X = torch.tensor([-1.1, 2.5], requires_grad=True)
    opt = tz.Modular([X], tz.m.GaussNewton(), tz.m.Backtracking())

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
    opt = tz.Modular(
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
    def __init__(self, reg:float = 1e-8, batched:bool=True, ):
        super().__init__(defaults=dict(batched=batched, reg=reg))

    @torch.no_grad
    def update(self, var):
        params = var.params
        batched = self.defaults['batched']

        closure = var.closure
        assert closure is not None

        # gauss newton direction
        with torch.enable_grad():
            f = var.get_loss(backward=False) # n_out
            assert isinstance(f, torch.Tensor)
            G_list = jacobian_wrt([f.ravel()], params, batched=batched)

        var.loss = f.pow(2).sum()

        G = self.global_state["G"] = flatten_jacobian(G_list) # (n_out, ndim)
        Gtf = G.T @ f.detach() # (ndim)
        self.global_state["Gtf"] = Gtf
        var.grad = vec_to_tensors(Gtf, var.params)

        # set closure to calculate sum of squares for line searches etc
        if var.closure is not None:
            def sos_closure(backward=True):
                if backward:
                    var.zero_grad()
                    with torch.enable_grad():
                        loss = closure(False).pow(2).sum()
                        loss.backward()
                    return loss

                loss = closure(False).pow(2).sum()
                return loss

            var.closure = sos_closure

    @torch.no_grad
    def apply(self, var):
        reg = self.defaults['reg']

        G = self.global_state['G']
        Gtf = self.global_state['Gtf']

        GtG = G.T @ G # (ndim, ndim)
        if reg != 0:
            GtG.add_(torch.eye(GtG.size(0), device=GtG.device, dtype=GtG.dtype).mul_(reg))

        v = torch.linalg.lstsq(GtG, Gtf).solution # pylint:disable=not-callable

        var.update = vec_to_tensors(v, var.params)
        return var

    def get_H(self, var):
        G = self.global_state['G']
        return linear_operator.AtA(G)
