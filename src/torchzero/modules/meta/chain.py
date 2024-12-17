import typing as T
from collections import abc

import torch

from ...core import OptimizerModule, ClosureType, OptimizationState
from ...tensorlist import TensorList
from .set_grad import ReturnAscent, SetGrad


class Chain(OptimizerModule):
    def __init__(self, modules: OptimizerModule | abc.Iterable[OptimizerModule]):
        """Chains multiple modules together.

        Note:
            The last module will return the ascent direction, which is passed to the child.
        """
        super().__init__({})
        if isinstance(modules, OptimizerModule): modules = [modules]
        else: modules = list(modules)

        # first module is chain's child, second module is first module's child, etc
        if len(modules) != 0:
            self._add_child_(modules[0])
            if len(modules) > 1:
                for i, m in enumerate(modules[:-1]):
                    m._set_next_module(modules[i+1])

            modules[-1]._set_next_module(ReturnAscent())

        self._chain_modules = modules

    @torch.no_grad
    def step(self, state: OptimizationState):
        if len(self.children) == 0: return
        ascent: TensorList = self.children[0].step(state)  # type:ignore
        state.ascent = ascent
        return self._update_params_or_step_with_next(state)

class ChainReturn(OptimizerModule):
    def __init__(self, modules: OptimizerModule | abc.Iterable[OptimizerModule]):
        """Chains multiple modules together. This must be last and will return whatever that last module returns,
        which is useful for modules like ReturnAscent."""
        super().__init__({})
        if isinstance(modules, OptimizerModule): modules = [modules]
        else: modules = list(modules)

        # first module is chain's child, second module is first module's child, etc
        if len(modules) != 0:
            self._add_child_(modules[0])
            if len(modules) > 1:
                for i, m in enumerate(modules[:-1]):
                    m._set_next_module(modules[i+1])

        self._chain_modules = modules

    @torch.no_grad
    def step(self, state: OptimizationState):
        if len(self.children) == 0: return
        return self.children[0].step(state)  # type:ignore

    # def _add_child_(self, child: "OptimizerModule"):
    #     """Add a child and initialize it's params."""
    #     self._chain_modules[-1]._add_child_(child)

    def _set_next_module(self, next_module: "OptimizerModule"):
        raise ValueError("LastChain must be last.")

