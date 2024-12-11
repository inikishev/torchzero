from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

def sign_grad_(params: abc.Iterable[torch.Tensor]):
    """Applies the sign function to the gradients of the given parameters."""
    TensorList(params).get_existing_grads().sign_()

class Sign(OptimizerModule):
    """Takes the sign of the ascent direction."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent):
        ascent.sign_()
        return ascent
