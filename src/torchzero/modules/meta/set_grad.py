import torch
from ...tensorlist import TensorList
from ...core import OptimizerModule, _get_loss, ClosureType

class SetGrad(OptimizerModule):
    """Sets gradient to ascent direction."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def step(self, state):
        if self.next_module is not None: raise ValueError("SetGrad can't have children")
        params = self.get_params()
        ascent_direction = state.maybe_use_grad_(params) # this will execute the closure which might be modified
        params.set_grad_(ascent_direction)
        return state.get_loss()

class ReturnAscent(OptimizerModule):
    """Step method returns the ascent direction which you can now manually subtract from your parameters."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def step(self, state) -> TensorList: # type:ignore
        if self.next_module is not None: raise ValueError("ReturnAscent can't have children")
        params = self.get_params()
        ascent_direction = state.maybe_use_grad_(params) # this will execute the closure which might be modified
        return ascent_direction

class ReturnClosure(OptimizerModule):
    def __init__(self):
        """Only works after modules that modify the closure.
        Step method returns the modified closure that you can pass to some pytorch optimizer."""
        super().__init__({})

    @torch.no_grad
    def step(self, state) -> ClosureType: # type:ignore
        if self.next_module is not None: raise ValueError("SetGrad can't have children")
        if state.closure is None:
            raise ValueError("MakeClosure requires closure")
        return state.closure

