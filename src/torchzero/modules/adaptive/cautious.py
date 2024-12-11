from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

class Cautious(OptimizerModule):
    def __init__(self, normalize = True, eps=1e-6):
        """
        Negates update for parameters where ascent direction and gradient sign is inconsistent.
        Also normalizes ascent direction by the number of parameters that are not masked.
        This is meant to be used after any momentum-based modules.

        This method has been described in *Cautious Optimizers: Improving Training with One Line of Code.
        Kaizhao Liang, Lizhang Chen, Bo Liu, Qiang Liu*

        Note:
            If you use this after modules that estimate gradients, e.g. FDM,
            hey need to have `make_closure` set to True so that they write to `grad` attribute.

        """
        super().__init__({})
        self.eps = eps
        self.normalize = normalize

    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()
        grad = state.maybe_compute_grad_(params)

        # mask will be > 0 for parameters where both signs are the same
        mask = (ascent * grad) > 0
        if self.normalize:
            mask = mask.to(ascent.dtype[0])
            mask /= mask.total_mean() + self.eps
        ascent *= mask

        return ascent

