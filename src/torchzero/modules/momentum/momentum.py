from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

def _heavyball_step(ascent, velocity: TensorList, momentum, dampening: TensorList):
    velocity.mul_(momentum).add_(ascent * (1 - dampening))
    return velocity.clone()

class HeavyBall(OptimizerModule):
    """Polyak's (heavy ball) momentum. Exactly matches pytorch SGD `momentum` option.

    Args:
        decay (float, optional): momentum decay. Defaults to 0.9.
        dampening (float, optional): momentum dampening. Defaults to 0.
    """
    def __init__(self, momentum: float = 0.9, dampening: float = 0, ):
        defaults = dict(momentum = momentum, dampening = dampening)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, vars, ascent):
        velocity = self.get_state_key('velocity', init = ascent)
        settings = self.get_all_group_keys()
        updated_direction = _heavyball_step(ascent, velocity, settings['momentum'], settings['dampening'])
        return updated_direction


def _nesterov_step_(ascent, velocity: TensorList, momentum, dampening,):
    # update velocity with the ascent direction
    velocity += ascent

    # decay velocity (this can be moved before previous line for slightly different results)
    velocity *= momentum

    # update ascent direction with velocity
    ascent += velocity * (1 - dampening)


class NesterovMomentum(OptimizerModule):
    """Nesterov momentum. Exactly matches pytorch SGD with `nesterov=True`,
    except this also supports dampening.

    Args:
        decay (float, optional): momentum decay. Defaults to 0.9.
        dampening (float, optional): momentum dampening. Defaults to 0.
    """
    def __init__(self, decay: float = 0.9, dampening: float = 0, ):
        defaults = dict(momentum = decay, dampening = dampening)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, vars, ascent):
        velocity = self.get_state_key('velocity')
        settings = self.get_all_group_keys()
        _nesterov_step_(ascent, velocity, settings['momentum'], settings['dampening'])
        return ascent

class GradientAveraging(OptimizerModule):
    """Averages last 2 gradients (TODO)"""
    def __init__(self, dampening: float = 0, ):
        defaults = dict(dampening = dampening)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, vars, ascent):
        velocity = self.get_state_key('velocity')
        dampening = self.get_group_key('dampening')

        new_direction = ascent + velocity * (1-dampening)
        velocity.copy_(ascent)

        return new_direction


class RandomCoordinateMomentum(OptimizerModule):
    """Only uses `p` random coordinates of the new update. Other coordinates remain from previous update.
    This works but I don't know if it is any good.

    Args:
        p (float, optional): probability to update velocity with a new weigh value. Defaults to 0.1.
        nesterov (bool, optional): if False, update uses delayed momentum. Defaults to True.
    """
    def __init__(self, p: float = 0.1, nesterov=True):
        defaults = dict(p=p)
        super().__init__(defaults)
        self.nesterov = nesterov

    @torch.no_grad
    def _update(self, vars, ascent):
        velocity = self.get_state_key('velocity', init = ascent)
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
