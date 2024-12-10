from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

class Cautious(OptimizerModule):
    def __init__(self, normalize = True, eps=1e-6):
        """COMPLETELY UNTESTED AND MAY NOT WORK AT ALL!!!
        Negates update for parameters where ascent direction and gradient sign is inconsistent.
        Otherwise ascent direction is squared (or am I misunderstanding it?).
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
        mask = ascent_direction * (grad > 0)
        if self.normalize: mask /= mask.total_mean() + self.eps
        ascent_direction *= mask

        return ascent_direction

class NegateOnSignInconsistence(OptimizerModule):
    def __init__(self, normalize = True, eps=1e-6):
        """COMPLETELY UNTESTED AND MAY NOT WORK AT ALL!!!
        Negates update for parameters where ascent direction and gradient sign is inconsistent.
        Also normalizes ascent direction by the number of parameters that are not masked.
        This is meant to be used after any momentum-based modules.

        This is same as cautious optimizers but without squaring
        *Cautious Optimizers: Improving Training with One Line of Code.
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
        grad_sign = state.maybe_compute_grad_(params).sign()

        # mask will be > 0 for parameters where both signs are the same
        mask = ascent_direction.sign() == grad_sign
        if self.normalize:
            mask = mask.to(ascent_direction.dtype[0])
            mask /= mask.total_mean() + self.eps
        ascent_direction *= mask

        return ascent_direction

class NegateOnSignChange(OptimizerModule):
    # todo: add momentum to negation (to cautious as well and rprop negation as well)
    def __init__(self, normalize = True, eps=1e-6, use_grad = False):
        """COMPLETELY UNTESTED AND MAY NOT WORK AT ALL!!!
        Negates update for parameters where where gradient or update sign changes.

        Note:
            If you use this after modules that estimate gradients, e.g. FDM,
            hey need to have `make_closure` set to True so that they write to `grad` attribute.

        """
        super().__init__({})
        self.eps = eps
        self.normalize = normalize
        self.use_grad = use_grad
        self.current_step = 0

    @torch.no_grad
    def _update(self, state, ascent_direction):
        params = self.get_params()
        if self.use_grad: cur_sign = state.maybe_compute_grad_(params).sign()
        else: cur_sign = ascent_direction.sign()
        prev_sign = self.get_group_key('prev_sign')

        # initialize on first step
        if self.current_step == 0:
            prev_sign.set_(cur_sign)
            return ascent_direction

        # mask will be > 0 for parameters where both signs are the same
        mask = ascent_direction.sign() == prev_sign
        if self.normalize:
            mask = mask.to(ascent_direction.dtype[0])
            mask /= mask.total_mean() + self.eps
        ascent_direction *= mask

        prev_sign.set_(cur_sign)
        self.current_step += 1
        return ascent_direction

class UndoOnSignChange(OptimizerModule):
    def __init__(self):
        """COMPLETELY UNTESTED AND MAY NOT WORK AT ALL!!!
        Undoes previous update for parameters where where gradient or update sign changes.

        Note:
            If you use this after modules that estimate gradients, e.g. FDM,
            hey need to have `make_closure` set to True so that they write to `grad` attribute.

        """
        super().__init__({})
        self.current_step = 0

    @torch.no_grad
    def _update(self, state, ascent_direction):
        cur_sign = ascent_direction.sign()
        prev_sign, prev_ascent = self.get_group_keys(['prev_sign', 'prev_update'])

        # initialize on first step
        if self.current_step == 0:
            prev_sign.set_(cur_sign)
            prev_ascent.copy_(ascent_direction)
            return ascent_direction

        ascent_direction -= prev_ascent.mul_(cur_sign != prev_sign)

        prev_sign.set_(cur_sign)
        prev_ascent.copy_(ascent_direction)
        self.current_step += 1
        return ascent_direction
