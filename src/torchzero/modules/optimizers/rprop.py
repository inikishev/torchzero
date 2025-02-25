from collections import abc

import torch

from ...tensorlist import TensorList, where
from ...core import OptimizerModule


def _bool_ones_like(x):
    return torch.ones_like(x, dtype=torch.bool)

class Rprop(OptimizerModule):
    """
    Resilient propagation. The update magnitude gets multiplied by `nplus` if gradient didn't change the sign,
    or `nminus` if it did. Then the update is applied with the sign of the current gradient.

    Additionally, if gradient changes sign, the update for that weight is reverted.
    Next step, magnitude for that weight won't change.

    Compared to pytorch this also implements backtracking update when sign changes.
    To make this behave exactly the same as `torch.optim.Rprop`, set `backtrack` to False.

    Args:
        nplus (float): multiplicative increase factor for when ascent didn't change sign (default: 1.2).
        nminus (float): multiplicative decrease factor for when ascent changed sign (default: 0.5).
        lb (float): minimum step size, can be None (default: 1e-6)
        ub (float): maximum step size, can be None (default: 50)
        backtrack (float):
            if True, when ascent sign changes, undoes last weight update, otherwise sets update to 0.
            When this is False, this exactly matches pytorch Rprop. (default: True)
        alpha (float): learning rate (default: 1).

    reference
        *Riedmiller, M., & Braun, H. (1993, March). A direct adaptive method for faster backpropagation learning:
        The RPROP algorithm. In IEEE international conference on neural networks (pp. 586-591). IEEE.*
    """
    def __init__(
        self,
        nplus: float = 1.2,
        nminus: float = 0.5,
        lb: float | None = 1e-6,
        ub: float | None = 50,
        backtrack=True,
        alpha: float = 1,
    ):
        defaults = dict(nplus = nplus, nminus = nminus, alpha = alpha, lb = lb, ub = ub)
        super().__init__(defaults)
        self.current_step = 0
        self.backtrack = backtrack

    @torch.no_grad
    def _update(self, vars, ascent):
        params = self.get_params()

        sign = ascent.sign_()
        nplus, nminus, lb, ub = self.get_group_keys('nplus', 'nminus', 'lb', 'ub')
        prev, allowed, magnitudes = self.get_state_keys(
            'prev', 'allowed', 'magnitudes',
            inits = [torch.zeros_like, _bool_ones_like, torch.zeros_like],
            params=params
        )

        # initialize on 1st step
        if self.current_step == 0:
            magnitudes.fill_(self.get_group_key('alpha')).clamp_(lb, ub)
            ascent = magnitudes * sign
            prev.copy_(ascent)
            self.current_step += 1
            return ascent

        mul = (sign * prev).mul_(allowed)

        sign_changed = mul < 0
        sign_same = mul > 0
        zeroes = mul == 0

        mul.fill_(1)
        mul.masked_fill_(sign_changed, nminus)
        mul.masked_fill_(sign_same, nplus)

        # multiply magnitudes based on sign change and clamp to bounds
        magnitudes.mul_(mul).clamp_(lb, ub)

        # revert update if sign changed
        if self.backtrack:
            ascent = sign.mul_(magnitudes)
            ascent.masked_set_(sign_changed, prev.neg_())
        else:
            ascent = sign.mul_(magnitudes * ~sign_changed)

        # update allowed to only have weights where last update wasn't reverted
        allowed.set_(sign_same | zeroes)

        prev.copy_(ascent)
        self.current_step += 1
        return ascent



