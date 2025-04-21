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