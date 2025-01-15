import typing
import torch

from ...core import OptimizerModule

class Cautious(OptimizerModule):
    """Negates update for parameters where update and gradient sign is inconsistent.
    Optionally normalizes the update by the number of parameters that are not masked.
    This is meant to be used after any momentum-based modules.

    Args:
        normalize (bool, optional):
            renormalize update after masking.
            only has effect when mode is 'zero'. Defaults to False.
        eps (float, optional): epsilon for normalization. Defaults to 1e-6.
        mode (str, optional):
            what to do with updates with inconsistent signs.

            "zero" - set them to zero (as in paper)

            "grad" - set them to the gradient

            "negate" - negate them (same as using update magnitude and gradient sign)

    .. warning::
        If you use this after modules that estimate gradients, e.g. FDM,
        hey need to have `make_closure` set to True so that they write to `grad` attribute.

    reference
        *Cautious Optimizers: Improving Training with One Line of Code.
        Kaizhao Liang, Lizhang Chen, Bo Liu, Qiang Liu*
    """
    def __init__(self, normalize = False, eps=1e-6, mode: typing.Literal['zero', 'grad', 'backtrack'] = 'zero'):
        super().__init__({})
        self.eps = eps
        self.normalize = normalize
        self.mode: typing.Literal['zero', 'grad', 'backtrack'] = mode

    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()
        grad = state.maybe_compute_grad_(params)

        # mask will be > 0 for parameters where both signs are the same
        mask = (ascent * grad) > 0
        if self.mode in ('zero', 'grad'):
            if self.normalize and self.mode == 'zero':
                fmask = mask.to(ascent[0].dtype)
                fmask /= fmask.total_mean() + self.eps
            else:
                fmask = mask

            ascent *= fmask

            if self.mode == 'grad':
                ascent += grad * mask.logical_not_()

            return ascent

        # mode = 'backtrack'
        ascent -= ascent.mul(2).mul_(mask.logical_not_())
        return ascent


class UseGradSign(OptimizerModule):
    """
    Uses update magnitude but gradient sign.

    .. warning::
        If `use_grad` is True and you use this after modules that estimate gradients, e.g. FDM,
        they need to have `make_closure` set to True so that they write to `grad` attribute.
    """
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()
        grad = state.maybe_compute_grad_(params)

        return ascent.abs_().mul_(grad.sign())

class UseGradMagnitude(OptimizerModule):
    """
    Uses update sign but gradient magnitude.

    .. warning::
        If `use_grad` is True and you use this after modules that estimate gradients, e.g. FDM,
        they need to have `make_closure` set to True so that they write to `grad` attribute.
    """
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()
        grad = state.maybe_compute_grad_(params)

        return ascent.sign_().mul_(grad.abs())


class ScaleLRBySignChange(OptimizerModule):
    """
    learning rate gets multiplied by `nplus` if ascent/gradient didn't change the sign,
    or `nminus` if it did.

    This is part of RProp update rule.

    Args:
        nplus (float): learning rate gets multiplied by `nplus` if ascent/gradient didn't change the sign
        nminus (float): learning rate gets multiplied by `nminus` if ascent/gradient changed the sign
        lb (float): lower bound for lr.
        ub (float): upper bound for lr.
        alpha (float): initial learning rate.

    .. warning::
        If `use_grad` is True and you use this after modules that estimate gradients, e.g. FDM,
        they need to have `make_closure` set to True so that they write to `grad` attribute.
    """
    def __init__(self, nplus: float = 1.2, nminus: float = 0.5, lb = 1e-6, ub = 50, alpha=1, use_grad=False):
        defaults = dict(nplus = nplus, nminus = nminus, alpha = alpha, lb = lb, ub = ub)
        super().__init__(defaults)
        self.current_step = 0
        self.use_grad = use_grad

    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()

        if self.use_grad: cur = state.maybe_compute_grad_(params)
        else: cur = ascent

        nplus, nminus, lb, ub = self.get_group_keys('nplus', 'nminus', 'lb', 'ub')
        prev, lrs = self.get_state_keys('prev_ascent', 'lrs', params=params)

        # initialize on 1st step
        if self.current_step == 0:
            lrs.fill_(self.get_group_key('alpha'))
            ascent.mul_(lrs)
            prev.copy_(ascent)
            self.current_step += 1
            return ascent

        mask = cur * prev
        sign_changed = mask < 0
        sign_same = mask > 0

        # multiply magnitudes where sign didn't change
        lrs.masked_set_(sign_same, lrs * nplus)
        # multiply magnitudes where sign changed
        lrs.masked_set_(sign_changed, lrs * nminus)
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
        if self.backtrack: ascent.masked_set_(mask, prev)
        else: ascent.select_set_(mask, 0)

        prev.set_(cur)
        self.current_step += 1
        return ascent
