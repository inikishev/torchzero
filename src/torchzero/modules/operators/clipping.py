from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule


def clip_grad_value_(params: abc.Iterable[torch.Tensor], value:float = 1):
    """Clip the gradients of an iterable of parameters at specified value."""
    TensorList(params).get_existing_grads().clamp_(-value, value)

class ClipValue(OptimizerModule):
    """Clip the ascent direction at specified value."""
    def __init__(self, value: float):
        defaults = dict(value = value)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent):
        value = self.get_group_key('value')
        ascent.clamp_(-value, value)
        return ascent

def clip_grad_norm_(params: abc.Iterable[torch.Tensor], max_norm: float, ord:float=2):
    """Rescales gradients to the given norm if they exceed it."""
    gradients = TensorList(params).get_existing_grads()
    norm = gradients.total_vector_norm(ord)
    if norm > max_norm:
        gradients.div_(norm / max_norm)

class ClipNorm(OptimizerModule):
    """Rescales ascent direction to the given norm if it exceeds it."""
    def __init__(self, max_norm: float, ord:float=2):
        super().__init__({})
        self.max_norm = max_norm
        self.ord = ord

    @torch.no_grad
    def _update(self, state, ascent):
        norm = ascent.total_vector_norm(self.ord)
        if norm > self.max_norm:
            ascent.div_(norm / self.max_norm)
        return ascent


