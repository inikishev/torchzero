from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule


def _rmsprop_step(ascent: TensorList, mean_sqr: TensorList, alpha, eps: TensorList):
    mean_sqr.mul_(alpha).addcmul_(ascent, ascent, value = 1 - alpha)
    return ascent.div_(mean_sqr.sqrt().add_(eps))

class RMSProp(OptimizerModule):
    def __init__(self, alpha: float = 0.99, eps: float = 1e-8):
        """https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

        Exactly matches pytorch RMSProp.

        Args:
            alpha (float, optional): _description_. Defaults to 0.99.
            eps (float, optional): _description_. Defaults to 1e-8.
        """
        defaults = dict(alpha = alpha, eps = eps)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent):
        mean_sqr = self.get_state_key('mean_sqr')
        settings = self.get_all_group_keys()
        updated_direction = _rmsprop_step(ascent, mean_sqr, settings['alpha'], settings['eps'])
        return updated_direction
