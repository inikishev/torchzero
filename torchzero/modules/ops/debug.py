from collections import deque

import torch

from ...core import Module
from ...utils.tensorlist import Distributions

class PrintUpdate(Module):
    def __init__(self, text = 'update = ', print_fn = print):
        defaults = dict(text=text, print_fn=print_fn)
        super().__init__(defaults)

    def step(self, vars):
        self.settings[vars.params[0]]["print_fn"](f'{self.settings[vars.params[0]]["text"]}{vars.update}')
        return vars

class PrintShape(Module):
    def __init__(self, text = 'shapes = ', print_fn = print):
        defaults = dict(text=text, print_fn=print_fn)
        super().__init__(defaults)

    def step(self, vars):
        shapes = [u.shape for u in vars.update] if vars.update is not None else None
        self.settings[vars.params[0]]["print_fn"](f'{self.settings[vars.params[0]]["text"]}{shapes}')
        return vars