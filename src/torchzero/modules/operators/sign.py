from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

def sign_grad_(params: abc.Iterable[torch.Tensor]):
    """Apply sign function to gradients of an iterable of parameters.

    Args:
        params (abc.Iterable[torch.Tensor]): an iterable of Tensors or a single Tensor.
    """
    TensorList(params).get_existing_grads().sign_()

class Sign(OptimizerModule):
    """Applies sign function to the update"""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent):
        ascent.sign_()
        return ascent
