from collections.abc import Iterable, Sequence
from typing import Any

import torch

from ...core import Chainable, Module


class Alternate(Module):
    """alternate between stepping with `modules`"""
    LOOP = True
    def __init__(self, *modules: Chainable, steps: int | Iterable[int] = 1):
        if isinstance(steps, Iterable):
            steps = list(steps)
            if len(steps) != len(modules):
                raise ValueError(f"steps must be the same length as modules, got {len(modules) = }, {len(steps) = }")

        defaults = dict(steps=steps)
        super().__init__(defaults)

        self.set_children_sequence(modules)
        self.global_state['current_module_idx'] = 0
        self.global_state['steps_to_next'] = steps[0] if isinstance(steps, list) else steps

    @torch.no_grad
    def step(self, var):
        # get current module
        current_module_idx = self.global_state.setdefault('current_module_idx', 0)
        module = self.children[f'module_{current_module_idx}']

        # step
        var = module.step(var.clone(clone_update=False))

        # number of steps until next module
        steps = self.settings[var.params[0]]['steps']
        if isinstance(steps, int): steps = [steps]*len(self.children)

        if 'steps_to_next' not in self.global_state:
            self.global_state['steps_to_next'] = steps[0] if isinstance(steps, list) else steps

        self.global_state['steps_to_next'] -= 1

        # switch to next module
        if self.global_state['steps_to_next'] == 0:
            self.global_state['current_module_idx'] += 1

            # loop to first module (or keep using last module on Switch)
            if self.global_state['current_module_idx'] > len(self.children) - 1:
                if self.LOOP: self.global_state['current_module_idx'] = 0
                else: self.global_state['current_module_idx'] = len(self.children) - 1

            self.global_state['steps_to_next'] = steps[self.global_state['current_module_idx']]

        return var

class Switch(Alternate):
    """switch to next module after some steps"""
    LOOP = False
    def __init__(self, *modules: Chainable, steps: int | Iterable[int]):

        if isinstance(steps, Iterable):
            steps = list(steps)
            if len(steps) != len(modules) - 1:
                raise ValueError(f"steps must be the same length as modules, got {len(modules) = }, {len(steps) = }")

            steps.append(1)

        super().__init__(*modules, steps=steps)