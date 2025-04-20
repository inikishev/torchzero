from collections import deque

import torch

from ...core import ParameterwiseTransform, Target, Transform, Module
from ...utils.tensorlist import Distributions

class PrintUpdate(Module):
    def __init__(self, text = 'update = ', print_fn = print):
        defaults = dict(text=text, print_fn=print_fn)
        super().__init__(defaults)

    def step(self, vars):
        self.defaults["print_fn"](f'{self.defaults["text"]}{vars.update}')
        return vars