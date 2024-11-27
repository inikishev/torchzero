from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

def normalize_grad_(params: abc.Iterable[torch.Tensor], norm_value:float = 1, ord:float = 2, min:float = 0, ):
    """Normalizes gradients to the given norm value."""
    gradients = TensorList(params).get_existing_grads()
    norm = gradients.total_vector_norm(ord)
    if norm > min:
        gradients.div_(norm / norm_value)

class Normalize(OptimizerModule):
    """Normalizes ascent direction to the given norm value."""
    def __init__(self, norm_value: float = 1, ord:float=2, min: float = 0):
        super().__init__({})
        self.norm_value = norm_value
        self.ord = ord
        self.min = min

    @torch.no_grad
    def _update(self, state, ascent_direction):
        norm = ascent_direction.total_vector_norm(self.ord)
        if norm > self.min:
            ascent_direction.div_(norm / self.norm_value)
        return ascent_direction
