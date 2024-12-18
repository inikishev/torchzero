import typing as T
from collections import abc

import torch

from ...core import OptimizerModule, OptimizationState


class Chain(OptimizerModule):
    """
    Utility module that chains multiple modules together, usually used by other modules.
    This must be last and will return whatever that last module returns.
    """
    def __init__(self, modules: OptimizerModule | abc.Iterable[OptimizerModule]):
        super().__init__({})
        if isinstance(modules, OptimizerModule): modules = [modules]
        else: modules = list(modules)

        # first module is chain's child, second module is first module's child, etc
        if len(modules) != 0:
            self._set_next_module(modules[0])
            if len(modules) > 1:
                for i, m in enumerate(modules[:-1]):
                    m._set_next_module(modules[i+1])

        self._chain_modules = modules

    @torch.no_grad
    def step(self, state: OptimizationState):
        return self._update_params_or_step_with_next(state)
