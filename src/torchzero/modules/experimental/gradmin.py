import torch

from ...core import OptimizerModule
from ...grad.derivatives import jacobian
from ...tensorlist import TensorList

class GradMin(OptimizerModule):
    """An idea.
    """
    def __init__(self, add_loss: float = 1, square=False, maximize_grad = False):
        super().__init__(dict(add_loss=add_loss))
        self.square = square
        self.maximize_grad = maximize_grad

    @torch.no_grad
    def step(self, state):
        if state.closure is None: raise ValueError()
        if state.ascent is not None:
            raise ValueError("GradMin doesn't accept ascent_direction")

        params = self.get_params()
        add_loss = self.get_group_key('add_loss')

        self.zero_grad()
        with torch.enable_grad():
            state.fx0 = state.closure(False)
            grads = jacobian([state.fx0], params, create_graph=True, batched=False) # type:ignore
            grads = TensorList(grads).squeeze_(0)
            if self.square:
                grads = grads ** 2
            else:
                grads = grads.abs()

            if self.maximize_grad: grads: TensorList = grads - (state.fx0 * add_loss) # type:ignore
            else: grads = grads + (state.fx0 * add_loss)
            grad_mean = torch.sum(torch.stack(grads.sum())) / grads.total_numel()
            grad_mean.backward(retain_graph=False)

        if self.maximize_grad: state.grad = params.ensure_grad_().grad.neg_()
        else: state.grad = params.ensure_grad_().grad

        state.maybe_use_grad_(params)
        return self._update_params_or_step_with_next(state)
