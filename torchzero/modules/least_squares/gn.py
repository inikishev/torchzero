import torch
from ...core import Module

from ...utils.derivatives import jacobian_wrt, flatten_jacobian
from ...utils import vec_to_tensors
from ...utils.linalg import linear_operator
class SumOfSquares(Module):
    """Sets loss to be the sum of squares of losses, this is meant to be used to test least squares methods against
    ordinary minimization methods."""
    def __init__(self):
        super().__init__()

    @torch.no_grad
    def step(self, var):
        closure = var.closure

        if closure is not None:
            def sos_closure(backward=True):
                if backward:
                    with torch.enable_grad():
                        loss = closure(False)
                        loss.pow(2).sum().backward()
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
    def __init__(self, reg:float = 1e-8, batched:bool=True, ):
        super().__init__(defaults=dict(batched=batched, reg=reg))

    @torch.no_grad
    def update(self, var):
        params = var.params
        setting = self.settings[params[0]]
        batched = setting['batched']

        closure = var.closure
        assert closure is not None

        # gauss newton direction
        with torch.enable_grad():
            f = var.get_loss(backward=False)
            assert isinstance(f, torch.Tensor)
            G_list = jacobian_wrt([f], params, batched=batched)

        self.global_state["f"] = f.detach()
        self.global_state["G"] = flatten_jacobian(G_list) # (n_out, ndim)

        # set closure to calculate sum of squares for line searches etc
        def sos_closure(backward=True):
            if backward:
                with torch.enable_grad():
                    loss = closure(False)
                    loss.pow(2).sum().backward()
                return loss

            loss = closure(False)
            return loss.pow(2).sum()

        var.closure = sos_closure

    @torch.no_grad
    def apply(self, var):
        params = var.params
        setting = self.settings[params[0]]
        reg = setting['reg']

        f = self.global_state['f']
        G = self.global_state['G']
        GtG = G.T @ G # (ndim, ndim)
        if reg != 0:
            GtG.add_(torch.eye(GtG.size(0), device=GtG.device, dtype=GtG.dtype).mul_(reg))

        Gtf = G.T @ f # (ndim)
        v = torch.linalg.lstsq(GtG, Gtf) # pylint:disable=not-callable

        var.update = vec_to_tensors(v, var.params)
        return var

    def get_B(self, var):
        G = self.global_state['G']
        return linear_operator.MtM(G)
