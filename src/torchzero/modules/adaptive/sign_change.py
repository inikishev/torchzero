from collections import abc

import torch

from ...tensorlist import TensorList, where
from ...core import OptimizerModule


class ScaleLRBySignChange(OptimizerModule):
    """
    learning rate gets multiplied by `nplus` if ascent/gradient didn't change the sign,
    or `nminus` if it did.

    This is part of RProp update rule.

    Args:
        lr (float): initial learning rate.
        nplus (float): learning rate gets multiplied by `nplus` if ascent/gradient didn't change the sign
        nminus (float): learning rate gets multiplied by `nminus` if ascent/gradient changed the sign
        lb (float): lower bound for lr.
        ub (float): upper bound for lr.

    .. warning::
        If `use_grad` is True and you use this after modules that estimate gradients, e.g. FDM,
        they need to have `make_closure` set to True so that they write to `grad` attribute.
    """
    def __init__(self, lr: float = 1, nplus: float = 1.2, nminus: float = 0.5, lb = 1e-6, ub = 50, use_grad=False):
        defaults = dict(nplus = nplus, nminus = nminus, lr = lr, lb = lb, ub = ub)
        super().__init__(defaults)
        self.current_step = 0
        self.use_grad = use_grad

    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()

        if self.use_grad: cur = ascent
        else: cur = state.maybe_compute_grad_(params)

        nplus, nminus, lb, ub = self.get_group_keys(['nplus', 'nminus', 'lb', 'ub'])
        prev, lrs = self.get_state_keys(['prev_ascent', 'lrs'], params=params)

        # initialize on 1st step
        if self.current_step == 0:
            lrs.fill_(self.defaults['lr'])
            ascent.mul_(lrs)
            prev.copy_(ascent)
            self.current_step += 1
            return ascent

        mask = cur * prev
        sign_changed = mask < 0
        sign_same = mask > 0

        # multiply magnitudes where sign didn't change
        lrs.select_set_(sign_same, lrs * nplus)
        # multiply magnitudes where sign changed
        lrs.select_set_(sign_changed, lrs * nminus)
        # bounds
        lrs.clamp_(lb, ub)

        ascent.mul_(lrs)
        prev.copy_(cur)
        self.current_step += 1
        return ascent



class NegateOnSignChange(OptimizerModule):
    """Negates or undoes update for parameters where where gradient or update sign changes.

    This is part of RProp update rule.

    Args:
        normalize (bool, optional): renormalize update after masking. Defaults to False.
        eps (_type_, optional): epsilon for normalization. Defaults to 1e-6.
        use_grad (bool, optional): if True, tracks sign change of the gradient,
            otherwise track sign change of the update. Defaults to True.
        backtrack (bool, optional): if True, undoes the update when sign changes, otherwise negates it.
            Defaults to True.

    .. warning::
        If `use_grad` is True and you use this after modules that estimate gradients, e.g. FDM,
        they need to have `make_closure` set to True so that they write to `grad` attribute.

    """
    # todo: add momentum to negation (to cautious as well and rprop negation as well)
    def __init__(self, normalize = False, eps=1e-6, use_grad = False, backtrack = True):
        super().__init__({})
        self.eps = eps
        self.normalize = normalize
        self.use_grad = use_grad
        self.backtrack = backtrack
        self.current_step = 0

    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()
        if self.use_grad: cur = state.maybe_compute_grad_(params)
        else: cur = ascent
        prev = self.get_state_key('prev')

        # initialize on first step
        if self.current_step == 0:
            prev.set_(cur)
            self.current_step += 1
            return ascent

        # mask will be > 0 for parameters where both signs are the same
        mask = (cur * prev) < 0
        if self.backtrack: ascent.select_set_(mask, prev)
        else: ascent.select_set_(mask, 0)

        prev.set_(cur)
        self.current_step += 1
        return ascent
