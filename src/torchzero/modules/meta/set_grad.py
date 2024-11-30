import typing as T
from collections import abc

import torch

from ...core import OptimizerModule, _get_loss
from .chain import Chain
class SetGrad(OptimizerModule):
    """Sets gradient to ascent direction."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def step(self, state):
        if self.child is not None: raise ValueError("SetGrad can't have children")
        params = self.get_params()
        ascent_direction = state.maybe_use_grad_(params) # this will execute the closure which might be modified
        params.set_grad_(ascent_direction)
        return state.get_loss()

class MakeClosure(OptimizerModule):
    def __init__(self):
        """Only works after modules that modify the closure. Step method returns the modified closure"""
        super().__init__({})

    @torch.no_grad
    def step(self, state): # type:ignore
        if self.child is not None: raise ValueError("SetGrad can't have children")
        if state.closure is None:
            raise ValueError("MakeClosure requires closure")
        return state.closure


class MakeClosureFromModules(Chain):
    def __init__(self, modules):
        """Step method returns the modified closure"""
        if isinstance(modules, OptimizerModule): modules = [modules]
        else: modules = list(modules)
        modules.append(SetGrad())
        super().__init__(modules)

    @torch.no_grad
    def step(self, state): # type:ignore
        if state.closure is None: raise ValueError('MakeClosureFromModules requires a closure.')
        closure = state.closure # closure shouldn't reference state attribute because it can be changed

        def update_closure(backward = True, **k):
            if self.child is None: raise ValueError('MakeClosureFromModules needs a child')

            # on backward, update the ascent direction via stepping with the child
            # which ends on a `SetGrad`.
            if backward: return self.child.step(state)
            else: return closure(backward, **k)

        return update_closure