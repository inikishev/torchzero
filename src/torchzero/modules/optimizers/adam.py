from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule


class Adam(OptimizerModule):
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """Adam module.

        Args:
            lr (float, optional): learning rate.
                Use it instead of LR module, since it is part of the update rule for mu. Defaults to 1e-3.
            beta1 (float, optional): exponential decay rate of gradient moving average. Defaults to 0.9.
            beta2 (float, optional): exponential decay rate of squared gradient moving average. Defaults to 0.999.
            eps (float, optional): epsilon for numerical stability. Defaults to 1e-8.
        """
        defaults = dict(lr = lr, beta1=beta1, beta2=beta2, eps=eps)
        super().__init__(defaults)

        self.current_step = 0

    @torch.no_grad
    def _update(self, state, ascent_direction):
        settings = self.get_all_group_keys()
        mu, sigma = self.get_state_keys(['mu', 'sigma'], inits = ['params', torch.zeros_like])
        lr = settings['lr']
        beta1 = settings['beta1']
        beta2 = settings['beta2']
        eps = settings['eps']
        self.current_step += 1

        mu.mul_(beta1).add_(ascent_direction * (1 - beta1))
        sigma.mul_(beta2).add_(ascent_direction.pow(2) * (1 - beta2))
        mu.div_(1 - beta1**self.current_step)
        sigma.div_(1 - beta2**self.current_step)
        mu.div_(sigma.sqrt_().add_(eps)).mul_(-lr)
        return mu

