from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

class Cautious(OptimizerModule):
    def __init__(self, eps=1e-6):
        """
        Applies mask function based on the sign consistency of ascent direction and gradient.

        This method has been described in *Cautious Optimizers: Improving Training with One Line of Code.
        Kaizhao Liang, Lizhang Chen, Bo Liu, Qiang Liu*

        Note:
        If you use this after modules that estimate gradients, e.g. FDM, they need to have `make_closure` set to True.

        """
        super().__init__({})
        self.eps = eps

    @torch.no_grad
    def _update(self, state, ascent_direction):
        params = self.get_params()
        grad = state.maybe_compute_grad_(params)
        mask = (ascent_direction * grad > 0).to(grad.dtype)
        ascent_direction *= mask / (mask.total_mean() + self.eps)

        return ascent_direction
