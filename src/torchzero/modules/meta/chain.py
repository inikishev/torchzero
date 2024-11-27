import typing as T
from collections import abc

import torch

from ...core import OptimizerModule


class Chain(OptimizerModule):
    def __init__(self, modules: OptimizerModule | abc.Iterable[OptimizerModule]):
        """Chains multiple modules together."""
        super().__init__({})
        if isinstance(modules, OptimizerModule): modules = [modules]
        else: modules = list(modules)

        # first module is chain's child, second module is first module's child, etc
        if len(modules) != 0:
            self._set_child_(modules[0])
            if len(modules) > 1:
                for i, m in enumerate(modules[:-1]):
                    m._set_child_(modules[i+1])

    @torch.no_grad
    def step(self, state):
        if self.child is None: return
        return self.child.step(state)
