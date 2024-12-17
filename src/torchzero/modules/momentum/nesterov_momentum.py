from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

def _nesterov_step_(ascent, velocity: TensorList, momentum, dampening,):
    # update velocity with the ascent direction
    velocity += ascent

    # decay velocity (this can be moved before previous line for slightly different results)
    velocity *= momentum

    # update ascent direction with velocity
    ascent += velocity * (1 - dampening)


class NesterovMomentum(OptimizerModule):
    def __init__(self, decay: float = 0.9, dampening: float = 0, ):
        """Nesterov momentum.

        Args:
            decay (float, optional): momentum decay. Defaults to 0.9.
            dampening (float, optional): momentum dampening. Defaults to 0.
        """
        defaults = dict(momentum = decay, dampening = dampening)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent):
        velocity = self.get_state_key('velocity')
        settings = self.get_all_group_keys()
        _nesterov_step_(ascent, velocity, settings['momentum'], settings['dampening'])
        return ascent