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
        self.defaults["print_fn"](f'{self.defaults["text"]}{var.update}')
        return var

class PrintShape(Module):
    """Prints shapes of the update."""
    def __init__(self, text = 'shapes = ', print_fn = print):
        defaults = dict(text=text, print_fn=print_fn)
        super().__init__(defaults)

    def step(self, var):
        shapes = [u.shape for u in var.update] if var.update is not None else None
        self.defaults["print_fn"](f'{self.defaults["text"]}{shapes}')
        return var

class PrintParams(Module):
    """Prints current update."""
    def __init__(self, text = 'params = ', print_fn = print):
        defaults = dict(text=text, print_fn=print_fn)
        super().__init__(defaults)

    def step(self, var):
        self.defaults["print_fn"](f'{self.defaults["text"]}{var.params}')
        return var


class PrintLoss(Module):
    """Prints var.get_loss()."""
    def __init__(self, text = 'loss = ', print_fn = print):
        defaults = dict(text=text, print_fn=print_fn)
        super().__init__(defaults)

    def step(self, var):
        self.defaults["print_fn"](f'{self.defaults["text"]}{var.get_loss(False)}')
        return var
