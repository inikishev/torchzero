from collections import abc

import torch

from ...tensorlist import TensorList, where
from ...core import OptimizerModule


def _bool_ones_like(x):
    return torch.ones_like(x, dtype=torch.bool)
class Rprop(OptimizerModule):
    def __init__(self, lr: float = 1, nplus: float = 1.2, nminus: float = 0.5, lb = 1e-6, ub = 50, backtrack=True):
        """
        Resilient propagation. The update magnitude gets multiplied by `nplus` if gradient didn't change the sign,
        or `nminus` if it did. Then the update is applied with the sign of the current gradient.

        Additionally, if gradient changes sign, the update for that weight is reverted.
        Next step, magnitude for that weight won't change.

        Compared to pytorch this also implements backtracking update when sign changes.
        To make this behave exactly the same as `torch.optim.Rprop`, set `backtrack` to False.

        *Riedmiller, M., & Braun, H. (1993, March). A direct adaptive method for faster backpropagation learning: The RPROP algorithm. In IEEE international conference on neural networks (pp. 586-591). IEEE.*

        Args:
            lr (float): _description_
            nplus (float): _description_
            nminus (float): _description_
            lb (float): _description_
            ub (float): _description_

        Note:
            If `use_grad` is True and you use this after modules that estimate gradients, e.g. FDM,
            they need to have `make_closure` set to True so that they write to `grad` attribute.
        """
        defaults = dict(nplus = nplus, nminus = nminus, lr = lr, lb = lb, ub = ub)
        super().__init__(defaults)
        self.current_step = 0
        self.backtrack = backtrack

    @torch.no_grad
    def _update(self, state, ascent_direction):
        params = self.get_params()

        sign = ascent_direction.sign_()
        nplus, nminus, lb, ub = self.get_group_keys(['nplus', 'nminus', 'lb', 'ub'])
        prev, allowed, magnitudes = self.get_state_keys(
            ['prev_ascent', 'prevent_update', 'magnitudes'],
            inits = [torch.zeros_like, _bool_ones_like, torch.zeros_like],
            params=params
        )

        # initialize on 1st step
        if self.current_step == 0:
            magnitudes.fill_(self.defaults['lr'])
            ascent = magnitudes * sign
            prev.copy_(ascent)
            self.current_step += 1
            return ascent

        mask = (sign * prev).mul_(allowed)

        sign_changed = mask < 0
        sign_same = mask > 0
        zeroes = mask == 0

        # multiply magnitudes where sign didn't change
        magnitudes.select_set_(sign_same, magnitudes * nplus)
        # multiply magnitudes where sign changed
        magnitudes.select_set_(sign_changed, magnitudes * nminus)
        # bounds
        magnitudes.clamp_(lb, ub)

        # revert update if sign changed
        if self.backtrack:
            ascent = sign.mul_(magnitudes)
            ascent.select_set_(sign_changed, prev.neg_())
        else:
            ascent = sign.mul_(magnitudes * ~sign_changed)

        # update allowed to only have weights where last update wasn't reverted
        allowed.set_(sign_same | zeroes)

        prev.copy_(ascent)
        self.current_step += 1
        return ascent
