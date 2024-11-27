from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

def _nesterov_step_(ascent_direction, velocity: TensorList, momentum, dampening,):
    # update velocity with the ascent direction
    velocity += ascent_direction

    # decay velocity (this can be moved before previous line for slightly different results)
    velocity *= momentum

    # update ascent direction with velocity
    ascent_direction += velocity * (1 - dampening)


class NesterovMomentum(OptimizerModule):
    def __init__(self, decay: float = 0.9, dampening: float = 0, ):
        defaults = dict(momentum = decay, dampening = dampening)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent_direction):
        velocity = self.get_state_key('velocity')
        settings = self.get_all_group_keys()
        _nesterov_step_(ascent_direction, velocity, settings['momentum'], settings['dampening'])
        return ascent_direction