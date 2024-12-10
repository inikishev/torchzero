from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

class RProp(OptimizerModule):
    def __init__(self, nplus: float = 1.2, nminus: float = 0.5, d0: float = 1, lb = 1e-6, ub = 50, revert=True, ):
        """COMPLETELY UNTESTED AND MAY NOT WORK AT ALL!!!
        Resilient propagation. The update magnitude gets multiplied by `nplus` if gradient didn't change the sign,
        or `nminus` if it did. Then the update is applied with the sign of the current gradient.

        Additionally, if gradient changes sign, the update for that weight is reverted without affecting update magnitude.

        *Riedmiller, M., & Braun, H. (1993, March). A direct adaptive method for faster backpropagation learning: The RPROP algorithm. In IEEE international conference on neural networks (pp. 586-591). IEEE.*

        Args:
            nplus (float): _description_
            nminus (float): _description_
            d0 (float): _description_
            lb (float): _description_
            ub (float): _description_
            revert (bool, optional): _description_. Defaults to False.
            use_grad (bool, optional): _description_. Defaults to False.

        Note:
            If `use_grad` is True and you use this after modules that estimate gradients, e.g. FDM,
            they need to have `make_closure` set to True so that they write to `grad` attribute.
        """
        defaults = dict(nplus = nplus, nminus = nminus, d0 = d0, lb = lb, ub = ub)
        super().__init__(defaults)
        self.revert = revert
        self.current_step = 0

    @torch.no_grad
    def _update(self, state, ascent_direction):
        params = self.get_params()

        cur_sign = ascent_direction.sign()
        nplus, nminus, lb, ub = self.get_group_keys(['nplus', 'nminus', 'lb', 'ub'])
        prev_sign, magnitudes = self.get_state_keys(['prev_ascent_sign', 'magnitudes'], params=params)

        # initialize on first step
        if self.current_step == 0:
            prev_sign.set_(cur_sign)
            ascent_direction = magnitudes.fill_(self.get_group_key('d0')) * cur_sign
            if self.revert:
                prev_ascent = self.get_state_key('prev_update', params=params)
                prev_ascent.copy_(ascent_direction)
            return ascent_direction

        # update magnitudes
        sign_changed = cur_sign != prev_sign
        magnitudes.set_(magnitudes.where(sign_changed, magnitudes.mul(nminus), magnitudes.mul(nplus))).clamp_(lb, ub)

        # create ascent direction
        ascent_direction = magnitudes * cur_sign

        if self.revert: # this can also be momentumed
            prev_ascent = self.get_state_key('prev_ascent', params=params)
            ascent_direction.sub_(prev_ascent.mul_(sign_changed))
            prev_ascent.copy_(ascent_direction)

        prev_sign.set_(cur_sign)
        self.current_step += 1
        return ascent_direction
