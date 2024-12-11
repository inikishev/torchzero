from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

class _tensor_like:
    """_foreach_lerp_ doesn't accept list of scalars, so I have to convert them to tensors
    meaning I have to give them appropriate dtype and device."""
    __slots__ = ("device", "dtype")
    def __init__(self, tensor):
        self.device = tensor.device
        self.dtype = tensor.dtype
    def __call__(self, x): return torch.tensor(x, dtype = self.dtype, device = self.device)

def _rmsprop_step_(ascent: TensorList, mean_sqr: TensorList, alpha, eps: TensorList):
    mean_sqr.mul_(alpha).addcmul_(ascent, ascent, value = 1 - alpha)
    return ascent.div_(mean_sqr.sqrt().add_(eps))

def _centered_rmsprop_step_(ascent: TensorList, mean_sqr: TensorList, mean: TensorList, alpha, eps: TensorList):
    mean_sqr.mul_(alpha).addcmul_(ascent, ascent, value = 1 - alpha)
    mean.lerp_(ascent, 1-alpha.map(_tensor_like(ascent[0])))
    return ascent.div_(mean_sqr.addcmul(mean, mean, value=-1).sqrt_().add_(eps))

class RMSProp(OptimizerModule):
    def __init__(self, alpha: float = 0.99, eps: float = 1e-8, centered=False):
        """https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

        Exactly matches pytorch RMSProp.

        Args:
            alpha (float, optional): _description_. Defaults to 0.99.
            eps (float, optional): _description_. Defaults to 1e-8.
        """
        defaults = dict(alpha = alpha, eps = eps)
        super().__init__(defaults)
        self.centered = centered

    @torch.no_grad
    def _update(self, state, ascent):
        settings = self.get_all_group_keys()
        if self.centered:
            mean, mean_sqr = self.get_state_keys(('mean', 'mean_sqr'))
            updated_direction = _centered_rmsprop_step_(ascent, mean_sqr, mean, settings['alpha'], settings['eps'])
        else:
            mean_sqr = self.get_state_key('mean_sqr')
            updated_direction = _rmsprop_step_(ascent, mean_sqr, settings['alpha'], settings['eps'])
        return updated_direction
