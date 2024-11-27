from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

class Cautious(OptimizerModule):
    def __init__(self):
        """
        Zeroes updates for parameters where ascent direction (momentum) sign is inconsistent with the gradient sign.

        This method has been described in *Cautious Optimizers: Improving Training with One Line of Code.
        Kaizhao Liang, Lizhang Chen, Bo Liu, Qiang Liu*

        Note:
        If you use this after modules that estimate gradients, e.g. FDM, they need to have `make_closure` set to True.

        """
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent_direction):
        params = self.get_params()
        grad = state.maybe_compute_grad_(params)

        mask = grad.sign().eq(ascent_direction.sign())
        ascent_direction *= mask

        return ascent_direction
