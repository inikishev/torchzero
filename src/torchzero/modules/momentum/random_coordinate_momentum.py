from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

class RandomCoordinateMomentum(OptimizerModule):
    def __init__(self, p: float = 0.1, nesterov=True):
        """Only uses `p` random coordinates of the new update. Other coordinates remain from previous update.
        This works but I don't know if it is any good.

        Args:
            p (float, optional): probability to update velocity with a new weigh value. Defaults to 0.1.
            nesterov (bool, optional): if False, update uses delayed momentum. Defaults to True.
        """
        defaults = dict(p=p)
        super().__init__(defaults)
        self.nesterov = nesterov

    @torch.no_grad
    def _update(self, state, ascent):
        velocity = self.get_state_key('velocity', init = 'grad')
        settings = self.get_all_group_keys()

        # pick p veclocity indexes to update with the new ascent direction
        indexes = ascent.bernoulli_like(settings['p']).as_bool()

        if self.nesterov:
            # update the velocity at those indexes
            velocity.masked_set_(mask = indexes, value = ascent)
            return velocity.clone()

        new_ascent = velocity.clone()
        velocity.masked_set_(mask = indexes, value = ascent)
        return new_ascent