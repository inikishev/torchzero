import torch

from ...core import OptimizerModule

class SquaredGradientNormFDM(OptimizerModule):
    """Experimental (maybe don't use yet).

    Args:
        eps (float, optional): finite difference epsilon. Defaults to 1e-3.
    """
    def __init__(self, eps=1e-3):
        super().__init__(dict(eps=eps))

    @torch.no_grad
    def step(self, state):
        if state.closure is None: raise ValueError()
        if state.ascent is not None:
            raise ValueError("HvInvFDM doesn't accept ascent_direction")

        params = self.get_params()
        eps = self.get_group_key('eps')
        grad_fx0 = state.maybe_compute_grad_(params).clone()
        state.grad = grad_fx0 # set state grad to the cloned version, since it will be overwritten

        params += grad_fx0 * eps
        with torch.enable_grad(): _ = state.closure(True)

        params -= grad_fx0 * eps

        newton = grad_fx0 * ((grad_fx0 * eps) / (params.grad - grad_fx0))
        newton.nan_to_num_(0,0,0)

        state.ascent = newton
        return self._update_params_or_step_with_next(state)
