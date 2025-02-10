import torch
from ...tensorlist import TensorList
from ...core import OptimizerModule, _get_loss, _ClosureType

class SetGrad(OptimizerModule):
    """Doesn't update parameters, instead replaces all parameters `.grad` attribute with the current update.
    You can now step with any pytorch optimizer that utilises the `.grad` attribute."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def step(self, vars):
        if self.next_module is not None: raise ValueError("SetGrad can't have children")
        params = self.get_params()
        g = vars.maybe_use_grad_(params) # this may execute the closure which might be modified
        params.set_grad_(g)
        return vars.get_loss()


class ReturnAscent(OptimizerModule):
    """Doesn't update parameters, instead returns the update as a TensorList of tensors."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def step(self, vars) -> TensorList: # type:ignore
        if self.next_module is not None: raise ValueError("ReturnAscent can't have children")
        params = self.get_params()
        update = vars.maybe_use_grad_(params) # this will execute the closure which might be modified
        return update

class ReturnClosure(OptimizerModule):
    """Doesn't update parameters, instead returns the current modified closure.
    For example, if you put this after :code:`torchzero.modules.FDM(target = "closure")`,
    the closure will set `.grad` attribute to gradients approximated via finite difference.
    You can then pass that closure to something that requires closure like `torch.optim.LBFGS`."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def step(self, vars) -> _ClosureType: # type:ignore
        if self.next_module is not None: raise ValueError("ReturnClosure can't have children")
        if vars.closure is None:
            raise ValueError("MakeClosure requires closure")
        return vars.closure

