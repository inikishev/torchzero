from collections import abc

import torch

from ...core import OptimizerModule, OptimizationState
from ...tensorlist import TensorList
from ...python_tools import flatten



class _MaybeReturnAscent(OptimizerModule):
    def __init__(self):
        super().__init__({})
        self._return_ascent = False
        
    @torch.no_grad
    def step(self, state: OptimizationState):
        assert self.next_module is None, self.next_module
        
        if self._return_ascent:
            return state.ascent
        
        return self._update_params_or_step_with_next(state)
        
class Chain(OptimizerModule):
    """
    Utility module that chains multiple modules together, usually used by other modules.
    This must be last and will return whatever that last module returns.
    """
    def __init__(self, *modules: OptimizerModule | abc.Iterable[OptimizerModule]):
        super().__init__({})
        flat_modules: list[OptimizerModule] = flatten(modules)
        self._ascent_returner = _MaybeReturnAscent()
        flat_modules.append(self._ascent_returner)

        # first module is chain's child, second module is first module's child, etc
        self._set_child_('first', flat_modules[0])
        if len(flat_modules) > 1:
            for i, m in enumerate(flat_modules[:-1]):
                m._set_next_module(flat_modules[i+1])

        self._chain_modules = flat_modules

    @torch.no_grad
    def step(self, state: OptimizationState):
        # no next module, step with the child
        if self.next_module is None:
            self._ascent_returner._return_ascent = False 
            return self.children['first'].step(state)

        # return ascent and pass it to next module
        self._ascent_returner._return_ascent = True # type:ignore
        state.ascent = self.children['first'].step(state) # type:ignore
        
        return self._update_params_or_step_with_next(state)