from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

class GradientAveraging(OptimizerModule):
    """Averages last 2 gradients (TODO)"""
    def __init__(self, dampening: float = 0, ):
        defaults = dict(dampening = dampening)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent_direction):
        velocity = self.get_state_key('velocity')
        dampening = self.get_group_key('dampening')

        new_direction = ascent_direction + velocity * (1-dampening)
        velocity.copy_(ascent_direction)

        return new_direction
