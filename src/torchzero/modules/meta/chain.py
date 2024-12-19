import typing as T
from collections import abc

import torch

from ...core import OptimizerModule, OptimizationState
from ...python_tools import flatten

class Chain(OptimizerModule):
    """
    Utility module that chains multiple modules together, usually used by other modules.
    This must be last and will return whatever that last module returns.
    """
    def __init__(self, *modules: OptimizerModule | abc.Iterable[OptimizerModule]):
        super().__init__({})
        flat_modules: list[OptimizerModule] = flatten(modules)

        # first module is chain's child, second module is first module's child, etc
        if len(flat_modules) != 0:
            self._set_next_module(flat_modules[0])
            if len(flat_modules) > 1:
                for i, m in enumerate(flat_modules[:-1]):
                    m._set_next_module(flat_modules[i+1])

        self._chain_modules = flat_modules

    @torch.no_grad
    def step(self, state: OptimizationState):
        return self._update_params_or_step_with_next(state)
