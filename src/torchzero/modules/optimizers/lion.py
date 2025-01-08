import torch

from ...core import OptimizerModule
from ...tensorlist import TensorList


def _lion_step_(ascent: TensorList, ema: TensorList, beta1, beta2,):
    update = (ema * beta1).add_(ascent * (1 - beta1)).sign_()
    ema.lerp_compat_(ascent, 1-beta2)
    return update

class Lion(OptimizerModule):
    """evolved sign momentum https://arxiv.org/abs/2302.06675"""

    def __init__(self, beta1: float = 0.9, beta2: float = 0.99):
        defaults = dict(beta1=beta1, beta2=beta2)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent):
        beta1, beta2 = self.get_group_keys(('beta1', 'beta2'))
        ema = self.get_state_key('ema')
        return _lion_step_(ascent,ema,beta1,beta2)
        