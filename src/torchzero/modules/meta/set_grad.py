import torch
from ...tensorlist import TensorList
from ...core import OptimizerModule, _get_loss, ClosureType

class SetGrad(OptimizerModule):
    def __init__(self):
        """Doesn't update parameters, instead replaces all parameters `.grad` attribute with the current update.
        You can now step with any pytorch optimizer that utilises the `.grad` attribute."""
        super().__init__({})

    @torch.no_grad
    def step(self, state):
        if self.next_module is not None: raise ValueError("SetGrad can't have children")
        params = self.get_params()
        ascent_direction = state.maybe_use_grad_(params) # this will execute the closure which might be modified
        params.set_grad_(ascent_direction)
        return state.get_loss()

class ReturnAscent(OptimizerModule):
    def __init__(self):
        """Doesn't update parameters, instead returns the update as a list of tensors."""
        super().__init__({})

    @torch.no_grad
    def step(self, state) -> TensorList: # type:ignore
        if self.next_module is not None: raise ValueError("ReturnAscent can't have children")
        params = self.get_params()
        update = state.maybe_use_grad_(params) # this will execute the closure which might be modified
        return update

class ReturnClosure(OptimizerModule):
    def __init__(self):
        """Doesn't update parameters, instead returns the current modified closure.
        For example, if you put this after `FDM(make_closure=True)`, the closure will set `.grad` attribute
        to gradients approximated via finite differences. You can then pass that closure
        to something like torch.optim.LBFGS."""
        super().__init__({})

    @torch.no_grad
    def step(self, state) -> ClosureType: # type:ignore
        if self.next_module is not None: raise ValueError("SetGrad can't have children")
        if state.closure is None:
            raise ValueError("MakeClosure requires closure")
        return state.closure

