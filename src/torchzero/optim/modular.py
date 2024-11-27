from collections import abc
import torch

from ..core import OptimizerModule, TensorListOptimizer, OptimizationState
from ..modules import Chain

class ModularOptimizer(TensorListOptimizer):
    def __init__(self, params, modules: abc.Iterable[OptimizerModule] | OptimizerModule):

        if isinstance(params, torch.nn.Module):
            self.model = params
            params = list(params.parameters())
        else:
            self.model = None
            params = list(params)

        super().__init__(params, {})

        if isinstance(modules, OptimizerModule): modules = [modules]
        self.modules = list(modules)
        self.chain = Chain(self.modules)
        self.chain._initialize_(params)

    def step(self, closure=None): # type:ignore
        return self.chain.step(OptimizationState(closure, self.model))
