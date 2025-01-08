import typing as T

import torch

from ...core import OptimizerModule
from ..momentum.momentum import _heavyball_step, _nesterov_step_

class SGD(OptimizerModule):
    """Same as `torch.optim.SGD` but as an optimizer module. Exactly matches `torch.optim.SGD`, except
    nesterov momentum additionally supports dampening, and negative momentum is allowed.

    Args:
        lr (float, optional): learning rate. Defaults to 1e-3.
        momentum (float, optional): momentum. Defaults to 0.
        dampening (float, optional): momentum dampening. Defaults to 0.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.
    """
    def __init__(
        self,
        lr: float = 1,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ):

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,)
        super().__init__(defaults)
        self.nesterov = nesterov
        self.current_step = 0

    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()
        settings = self.get_all_group_keys()

        if any(i != 0 for i in settings['weight_decay']):
            ascent += params * settings['weight_decay']

        ascent *= settings['lr']

        if any(i != 0 for i in settings['momentum']):
            velocity = self.get_state_key('velocity', init = torch.zeros_like if self.nesterov else ascent)
            # consistency with pytorch which on first step only initializes momentum
            if self.current_step > 0 or self.nesterov:
                # nesterov step can be done in-place, polyak returns new direction
                if self.nesterov: _nesterov_step_(ascent, velocity, settings['momentum'], settings['dampening'])
                else: ascent = _heavyball_step(ascent, velocity, settings['momentum'], settings['dampening'])

        self.current_step += 1
        return ascent