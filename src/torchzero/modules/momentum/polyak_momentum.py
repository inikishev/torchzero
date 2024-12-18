from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

def _polyak_step(ascent, velocity: TensorList, momentum, dampening: TensorList):
    # add velocity to ascent direction
    updated_direction = ascent + velocity * (1 - dampening)

    # add ascent direction to velocity (one before the update!)
    velocity += ascent

    # decay velocity
    velocity *= momentum

    return updated_direction

class PolyakMomentum(OptimizerModule):
    """Polyak's (heavyball) momentum. Exactly matches pytorch SGD `momentum` option.

    Args:
        decay (float, optional): momentum decay. Defaults to 0.9.
        dampening (float, optional): momentum dampening. Defaults to 0.
    """
    def __init__(self, momentum: float = 0.9, dampening: float = 0, ):
        defaults = dict(momentum = momentum, dampening = dampening)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent):
        velocity = self.get_state_key('velocity')
        settings = self.get_all_group_keys()
        updated_direction = _polyak_step(ascent, velocity, settings['momentum'], settings['dampening'])
        return updated_direction
