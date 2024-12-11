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
    def _update(self, state, ascent_direction):
        params = self.get_params()
        grad = state.maybe_compute_grad_(params)

        # mask will be > 0 for parameters where both signs are the same
        mask = (ascent_direction * grad) > 0
        if self.normalize:
            mask = mask.to(ascent_direction.dtype[0])
            mask /= mask.total_mean() + self.eps
        ascent_direction *= mask

        return ascent_direction


class NegateOnSignChange(OptimizerModule):
    # todo: add momentum to negation (to cautious as well and rprop negation as well)
    def __init__(self, normalize = True, eps=1e-6, use_grad = True, backtrack = True):
        """
        Negates or undoes update for parameters where where gradient or update sign changes.

        Note:
            If you use this after modules that estimate gradients, e.g. FDM and `use_grad` is True,
            they need to have `make_closure` set to True so that they write to `grad` attribute.

        """
        super().__init__({})
        self.eps = eps
        self.normalize = normalize
        self.use_grad = use_grad
        self.backtrack = backtrack
        self.current_step = 0

    @torch.no_grad
    def _update(self, state, ascent_direction):
        params = self.get_params()
        if self.use_grad: cur = state.maybe_compute_grad_(params)
        else: cur = ascent_direction
        prev = self.get_state_key('prev')

        # initialize on first step
        if self.current_step == 0:
            prev.set_(cur)
            self.current_step += 1
            return ascent_direction

        # mask will be > 0 for parameters where both signs are the same
        mask = (cur * prev) < 0
        if self.backtrack: ascent_direction.select_set_(mask, prev) # type:ignore
        else: ascent_direction.select_set_(mask, 0) # type:ignore

        prev.set_(cur)
        self.current_step += 1
        return ascent_direction
