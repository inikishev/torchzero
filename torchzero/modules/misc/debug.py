from collections import deque

import torch

from ...core import Module
from ...utils.tensorlist import Distributions

class PrintUpdate(Module):
    """Prints current update."""
    def __init__(self, text = 'update = ', print_fn = print):
        defaults = dict(text=text, print_fn=print_fn)
        super().__init__(defaults)

    def step(self, var):
        self.settings[var.params[0]]["print_fn"](f'{self.settings[var.params[0]]["text"]}{var.update}')
        return var

class PrintShape(Module):
    """Prints shapes of the update."""
    def __init__(self, text = 'shapes = ', print_fn = print):
        defaults = dict(text=text, print_fn=print_fn)
        super().__init__(defaults)

    def step(self, var):
        shapes = [u.shape for u in var.update] if var.update is not None else None
        self.settings[var.params[0]]["print_fn"](f'{self.settings[var.params[0]]["text"]}{shapes}')
        return var

class PrintParams(Module):
    """Prints current update."""
    def __init__(self, text = 'params = ', print_fn = print):
        defaults = dict(text=text, print_fn=print_fn)
        super().__init__(defaults)

    def step(self, var):
        self.settings[var.params[0]]["print_fn"](f'{self.settings[var.params[0]]["text"]}{var.params}')
        return var


class PrintLoss(Module):
    """Prints var.get_loss()."""
    def __init__(self, text = 'loss = ', print_fn = print):
        defaults = dict(text=text, print_fn=print_fn)
        super().__init__(defaults)

    def step(self, var):
        self.settings[var.params[0]]["print_fn"](f'{self.settings[var.params[0]]["text"]}{var.get_loss(False)}')
        return var
