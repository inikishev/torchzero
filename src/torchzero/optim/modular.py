from collections import abc
import torch

from ..core import OptimizerModule, TensorListOptimizer, OptimizationState, Chain, _Chainable
from ..python_tools import flatten

class Modular(TensorListOptimizer):
    """Make a modular optimizer from a sequence of modules

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        modules (Iterable[OptimizerModule] | OptimizerModule):
            sequence of modules to chain together.
    """
    def __init__(self, params, *modules: _Chainable):
        flat_modules = flatten(modules)

        if isinstance(params, torch.nn.Module):
            self.model = params
            params = list(params.parameters())
        else:
            self.model = None
            params = list(params)

        super().__init__(params, {})

        self.modules = flat_modules
        self.chain = Chain(flat_modules)
        self.chain._initialize_(params)

    def step(self, closure=None): # type:ignore
        return self.chain.step(OptimizationState(closure, self.model))
