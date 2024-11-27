import typing as T

import torch

from ...core import OptimizerModule
from ..momentum.nesterov_momentum import _nesterov_step_
from ..momentum.polyak_momentum import _polyak_step


class SGD(OptimizerModule):
    """Same as torch.optim.SGD but as an optimizer module."""

    def __init__(
        self,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,)
        super().__init__(defaults)
        self.nesterov = nesterov

    @torch.no_grad
    def _update(self, state, ascent_direction):
        params = self.get_params()
        velocity = self.get_state_key('velocity')
        settings = self.get_all_group_keys()

        if any(i != 0 for i in settings['weight_decay']):
            ascent_direction += params * settings['weight_decay']

        ascent_direction *= settings['lr']

        if any(i != 0 for i in settings['momentum']):
            # nesterov step can be done in-place, polyak returns new direction
            if self.nesterov: _nesterov_step_(ascent_direction, velocity, settings['momentum'], settings['dampening'])
            else: ascent_direction = _polyak_step(ascent_direction, velocity, settings['momentum'], settings['dampening'])

        return ascent_direction