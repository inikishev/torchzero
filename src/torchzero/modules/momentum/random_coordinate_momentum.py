from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

class RandomCoordinateMomentum(OptimizerModule):
    def __init__(self, p: float = 0.1, decay: float = 0.99, dampening: float = 0, ):
        """Only uses `p` random coordinates of the new update. Other coordinates remain from previous update.
        This works but I don't know if it is any good."""
        defaults = dict(p=p, decay=decay, dampening = dampening)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent):
        velocity = self.get_state_key('velocity')
        settings = self.get_all_group_keys()

        # apply velocity
        updated_direction = ascent + velocity * (1 - settings['dampening'])

        # decay velocity
        velocity *= settings['decay']

        # we pick p indexes to update with the new ascent direction
        indexes = ascent.bernoulli_like(settings['p']).as_bool()

        # update the velocity at those indexes
        for a, v, i in zip(ascent, velocity, indexes):
            v[i] = a[i].clone()


        return updated_direction


class RandomCoordinateNesterovMomentum(OptimizerModule):
    def __init__(self, p: float = 0.1, decay: float = 0.99, dampening: float = 0, ):
        """Only uses `p` random coordinates of the new update. Other coordinates remain from previous update.
        This works but I don't know if it is any good."""
        defaults = dict(p=p, decay=decay, dampening = dampening)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent):
        velocity = self.get_state_key('velocity')
        settings = self.get_all_group_keys()

        # we pick p indexes to update with the new ascent direction
        indexes = ascent.bernoulli_like(settings['p']).as_bool()

        # update the velocity at those indexes
        # clone ascent direction so that it doesn't keep updating velocity with it
        for a, v, i in zip(ascent.clone(), velocity, indexes):
            v[i] = a[i]

        # decay velocity
        velocity *= settings['decay']

        # apply velocity
        ascent += velocity * (1 - settings['dampening'])

        return ascent
