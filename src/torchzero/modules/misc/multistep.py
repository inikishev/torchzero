from collections.abc import Callable, Iterable

import torch

from ...tensorlist import TensorList

from ...core import OptimizerModule, _Chainable


class Multistep(OptimizerModule):
    """Performs multiple steps (per batch), passes total update to the next module.

    Args:
        modules (_Chainable): modules to perform multiple steps with.
        num_steps (int, optional): number of steps to perform. Defaults to 2.
    """
    def __init__(self, modules: _Chainable, num_steps: int = 2):
        super().__init__({})
        self.num_steps = num_steps

        self._set_child_('modules', modules)

    def step(self, state):
        # no next module, just perform multiple steps
        if self.next_module is None:
            ret = None
            for step in range(self.num_steps):
                state_copy = state.copy(clone_ascent=True) if step != self.num_steps - 1 else state
                ret = self.children['modules'].step(state_copy)

                # since parameters are updated after stepping, grad and fx0 must be erased as they are no longer correct
                state.grad = None; state.fx0 = None

            return ret

        # accumulate steps and pass to next module
        p0 = self.get_params().clone()
        for step in range(self.num_steps):
            state_copy = state.copy(clone_ascent=True) if step != self.num_steps - 1 else state
            self.children['modules'].step(state_copy)

            # since parameters are updated after stepping, grad and fx0 must be erased as they are no longer correct
            state.grad = None; state.fx0 = None

        p1 = self.get_params()
        state.ascent = p0 - p1

        # undo ascent
        p1.set_(p0)

        return self._update_params_or_step_with_next(state, p1)