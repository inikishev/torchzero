from collections import deque

import torch

from ...core import Module
from ...utils.tensorlist import Distributions

class PrintUpdate(Module):
    def __init__(self, text = 'update = ', print_fn = print):
        defaults = dict(text=text, print_fn=print_fn)
        super().__init__(defaults)

    def step(self, var):
        self.settings[var.params[0]]["print_fn"](f'{self.settings[var.params[0]]["text"]}{var.update}')
        return var

class PrintShape(Module):
    def __init__(self, text = 'shapes = ', print_fn = print):
        defaults = dict(text=text, print_fn=print_fn)
        super().__init__(defaults)

    def step(self, var):
        shapes = [u.shape for u in var.update] if var.update is not None else None
        self.settings[var.params[0]]["print_fn"](f'{self.settings[var.params[0]]["text"]}{shapes}')
        return var