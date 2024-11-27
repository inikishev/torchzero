import torch

from ...core import OptimizerModule

class NewtonGradFDM(OptimizerModule):
    def __init__(self, eps=1e-3):
        """Performs a diagonal newton step the hessian
        approximated via finite difference two forwards and backwards from two gradients.

        Args:
            eps (float, optional): finite difference epsilon. Defaults to 1e-3.
        """
        super().__init__(dict(eps=eps))

    @torch.no_grad
    def step(self, state):
        if state.closure is None: raise ValueError()
        if state.ascent_direction is not None:
            raise ValueError("DiagNewtonViaHessianGradientProduct doesn't accept ascent_direction")

        params = self.get_params()
        eps = self.get_group_key('eps')
        grad_fx0 = state.maybe_compute_grad_(params).clone()
        state.grad = grad_fx0 # set state grad to the cloned version, since it will be overwritted

        params += grad_fx0 * eps
        with torch.enable_grad(): _ = state.closure(True)

        params -= grad_fx0 * eps

        newton = grad_fx0 * ((grad_fx0 * eps) / (params.grad - grad_fx0))
        newton.nan_to_num_(0,0,0)

        state.ascent_direction = newton
        return self._update_params_or_step_with_child(state)
