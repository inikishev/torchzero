from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule


def _rmsprop_step_(ascent: TensorList, mean_sqr: TensorList, smoothing, eps: TensorList):
    mean_sqr.mul_(smoothing).addcmul_(ascent, ascent, value = 1 - smoothing)
    return ascent.div_(mean_sqr.sqrt().add_(eps))

def _centered_rmsprop_step_(ascent: TensorList, mean_sqr: TensorList, mean: TensorList, smoothing, eps: TensorList):
    mean_sqr.mul_(smoothing).addcmul_(ascent, ascent, value = 1 - smoothing)
    mean.lerp_compat_(ascent, 1-smoothing)
    return ascent.div_(mean_sqr.addcmul(mean, mean, value=-1).sqrt_().add_(eps))

class RMSProp(OptimizerModule):
    """
    Divides ascent direction by running average of its mean square root.

    Exactly matches `torch.optim.RMSProp`.

    Args:
        smoothing (float, optional):
            smoothing constant (decay of ascent mean square root running average).
            Defaults to 0.99.
        eps (float, optional): term added to the denominator to improve numerical stability. Defaults to 1e-8.
        centered (float, optional):
            if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance.
            Defaults to False.

    reference
        https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    def __init__(self, smoothing: float = 0.99, eps: float = 1e-8, centered=False):

        defaults = dict(smoothing = smoothing, eps = eps)
        super().__init__(defaults)
        self.centered = centered

    @torch.no_grad
    def _update(self, state, ascent):
        settings = self.get_all_group_keys()
        if self.centered:
            mean, mean_sqr = self.get_state_keys('mean', 'mean_sqr')
            updated_direction = _centered_rmsprop_step_(ascent, mean_sqr, mean, settings['smoothing'], settings['eps'])
        else:
            mean_sqr = self.get_state_key('mean_sqr')
            updated_direction = _rmsprop_step_(ascent, mean_sqr, settings['smoothing'], settings['eps'])
        return updated_direction
