from collections import abc

import torch

from ...tensorlist import TensorList, where
from ...core import OptimizerModule


class UseGradSign(OptimizerModule):
    def __init__(self):
        """
        use ascent magnitude but gradient sign.

        Note:
            If `use_grad` is True and you use this after modules that estimate gradients, e.g. FDM,
            they need to have `make_closure` set to True so that they write to `grad` attribute.
        """
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()
        grad = state.maybe_compute_grad_(params)

        return ascent.abs_().mul_(grad.sign())

class UseGradMagnitude(OptimizerModule):
    def __init__(self):
        """
        use ascent sign but gradient magnitude.

        Note:
            If `use_grad` is True and you use this after modules that estimate gradients, e.g. FDM,
            they need to have `make_closure` set to True so that they write to `grad` attribute.
        """
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()
        grad = state.maybe_compute_grad_(params)

        return ascent.sign_().mul_(grad.abs())