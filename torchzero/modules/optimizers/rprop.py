
import torch

from ...core import Module, Target, Transform
from ...utils import NumberList, TensorList, as_tensorlist, unpack_dicts, unpack_states


def _bool_ones_like(x):
    return torch.ones_like(x, dtype=torch.bool)

def sign_consistency_lrs_(
    tensors: TensorList,
    prev_: TensorList,
    lrs_: TensorList,
    nplus: float | NumberList,
    nminus: float | NumberList,
    lb: float | NumberList,
    ub: float | NumberList,
    step: int,
):
    """returns `lrs_`"""
    sign = tensors.sign()
    if step == 0:
        prev_.set_(sign)
        return lrs_.clamp_(lb, ub)

    mul = sign * prev_
    prev_.set_(sign)

    sign_changed = mul < 0
    sign_same = mul > 0

    mul.fill_(1)
    mul.masked_fill_(sign_changed, nminus)
    mul.masked_fill_(sign_same, nplus)

    # multiply magnitudes based on sign change and clamp to bounds
    lrs_.mul_(mul).clamp_(lb, ub)
    return lrs_

def scale_by_sign_change_(
    tensors_: TensorList,
    cur: TensorList,
    prev_: TensorList,
    lrs_: TensorList,
    nplus: float | NumberList,
    nminus: float | NumberList,
    lb: float | NumberList,
    ub: float | NumberList,
    step: int,
):
    """returns `tensors_`"""
    lrs_ = sign_consistency_lrs_(cur,prev_=prev_,lrs_=lrs_,nplus=nplus,nminus=nminus,
                             lb=lb,ub=ub,step=step)
    return tensors_.mul_(lrs_)

def backtrack_on_sign_change_(
    tensors_: TensorList,
    cur: TensorList,
    prev_: TensorList,
    backtrack: bool,
    step: int
):
    """returns `tensors_`."""
    if step == 0:
        prev_.set_(cur)
        return tensors_

    # mask will be > 0 for parameters where both signs are the same
    mask = (cur * prev_) < 0
    if backtrack: tensors_.masked_set_(mask, prev_)
    else: tensors_.select_set_(mask, 0)

    prev_.set_(cur)
    return tensors_

def rprop_(
    tensors_: TensorList,
    prev_: TensorList,
    allowed_: TensorList,
    magnitudes_: TensorList,
    nplus: float | NumberList,
    nminus: float | NumberList,
    lb: float | NumberList,
    ub: float | NumberList,
    alpha: float | NumberList,
    backtrack: bool,
    step: int,
):
    """returns new tensors."""

    sign = tensors_.sign_()

    # initialize on 1st step
    if step == 0:
        magnitudes_.fill_(alpha).clamp_(lb, ub)
        new_tensors = magnitudes_ * sign
        prev_.copy_(new_tensors)
        return new_tensors

    mul = (sign * prev_).mul_(allowed_)

    sign_changed = mul < 0
    sign_same = mul > 0
    zeroes = mul == 0

    mul.fill_(1)
    mul.masked_fill_(sign_changed, nminus)
    mul.masked_fill_(sign_same, nplus)

    # multiply magnitudes based on sign change and clamp to bounds
    magnitudes_.mul_(mul).clamp_(lb, ub)

    # revert update if sign changed
    if backtrack:
        new_tensors = sign.mul_(magnitudes_)
        new_tensors.masked_set_(sign_changed, prev_.neg_())
    else:
        new_tensors = sign.mul_(magnitudes_ * ~sign_changed)

    # update allowed to only have weights where last update wasn't reverted
    allowed_.set_(sign_same | zeroes)

    prev_.copy_(new_tensors)
    return new_tensors



class Rprop(Transform):
    """
    Resilient propagation. The update magnitude gets multiplied by `nplus` if gradient didn't change the sign,
    or `nminus` if it did. Then the update is applied with the sign of the current gradient.

    Additionally, if gradient changes sign, the update for that weight is reverted.
    Next step, magnitude for that weight won't change.

    Compared to pytorch this also implements backtracking update when sign changes.

    This implementation is identical to :code:`torch.optim.Rprop` if :code:`backtrack` is set to False.

    Args:
        nplus (float): multiplicative increase factor for when ascent didn't change sign (default: 1.2).
        nminus (float): multiplicative decrease factor for when ascent changed sign (default: 0.5).
        lb (float): minimum step size, can be None (default: 1e-6)
        ub (float): maximum step size, can be None (default: 50)
        backtrack (float):
            if True, when ascent sign changes, undoes last weight update, otherwise sets update to 0.
            When this is False, this exactly matches pytorch Rprop. (default: True)
        alpha (float): initial per-parameter learning rate (default: 1).

    reference
        *Riedmiller, M., & Braun, H. (1993, March). A direct adaptive method for faster backpropagation learning:
        The RPROP algorithm. In IEEE international conference on neural networks (pp. 586-591). IEEE.*
    """
    def __init__(
        self,
        nplus: float = 1.2,
        nminus: float = 0.5,
        lb: float = 1e-6,
        ub: float = 50,
        backtrack=True,
        alpha: float = 1,
    ):
        defaults = dict(nplus = nplus, nminus = nminus, alpha = alpha, lb = lb, ub = ub, backtrack=backtrack)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        nplus, nminus, lb, ub, alpha = unpack_dicts(settings, 'nplus', 'nminus', 'lb', 'ub', 'alpha', cls=NumberList)
        prev, allowed, magnitudes = unpack_states(
            states, tensors,
            'prev','allowed','magnitudes',
            init=[torch.zeros_like, _bool_ones_like, torch.zeros_like],
            cls = TensorList,
        )

        tensors = rprop_(
            tensors_ = as_tensorlist(tensors),
            prev_ = prev,
            allowed_ = allowed,
            magnitudes_ = magnitudes,
            nplus = nplus,
            nminus = nminus,
            lb = lb,
            ub = ub,
            alpha = alpha,
            backtrack=settings[0]['backtrack'],
            step=step,
        )

        return tensors


class ScaleLRBySignChange(Transform):
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

    """

    def __init__(
        self,
        nplus: float = 1.2,
        nminus: float = 0.5,
        lb=1e-6,
        ub=50.0,
        alpha=1.0,
        use_grad=False,
        target: Target = "update",
    ):
        defaults = dict(nplus=nplus, nminus=nminus, alpha=alpha, lb=lb, ub=ub, use_grad=use_grad)
        super().__init__(defaults, uses_grad=use_grad, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        tensors = as_tensorlist(tensors)
        use_grad = settings[0]['use_grad']
        if use_grad: cur = as_tensorlist(grads)
        else: cur = tensors

        nplus, nminus, lb, ub = unpack_dicts(settings, 'nplus', 'nminus', 'lb', 'ub', cls=NumberList)
        prev, lrs = unpack_states(states, tensors, 'prev', 'lrs', cls=TensorList)

        if step == 0:
            lrs.set_(tensors.full_like([s['alpha'] for s in settings]))

        tensors = scale_by_sign_change_(
            tensors_ = tensors,
            cur = cur,
            prev_ = prev,
            lrs_ = lrs,
            nplus = nplus,
            nminus = nminus,
            lb = lb,
            ub = ub,
            step = step,
        )
        return tensors

class BacktrackOnSignChange(Transform):
    """Negates or undoes update for parameters where where gradient or update sign changes.

    This is part of RProp update rule.

    Args:
        normalize (bool, optional): renormalize update after masking. Defaults to False.
        eps (_type_, optional): epsilon for normalization. Defaults to 1e-6.
        use_grad (bool, optional):
            if True, tracks sign change of the gradient,
            otherwise track sign change of the update. Defaults to True.
        backtrack (bool, optional):
            if True, undoes the update when sign changes, otherwise negates it.
            Defaults to True.

    """
    def __init__(self, use_grad = False, backtrack = True, target: Target = 'update'):
        defaults = dict(use_grad=use_grad, backtrack=backtrack, target=target)
        super().__init__(defaults, uses_grad=use_grad)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        tensors = as_tensorlist(tensors)
        use_grad = settings[0]['use_grad']
        backtrack = settings[0]['backtrack']

        if use_grad: cur = as_tensorlist(grads)
        else: cur = tensors

        tensors = backtrack_on_sign_change_(
            tensors_ = tensors,
            cur = cur,
            prev_ = unpack_states(states, tensors, 'prev', cls=TensorList),
            backtrack = backtrack,
            step = step,
        )

        return tensors

class SignConsistencyMask(Transform):
    """
    Outputs a mask of sign consistency of current and previous inputs.

    The output is 0 for weights where input sign changed compared to previous input, 1 otherwise.

    Examples:

        GD that skips update for weights where gradient sign changed compared to previous gradient.

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.Mul(tz.m.SignConsistencyMask()),
                tz.m.LR(1e-2)
            )

    """
    def __init__(self,target: Target = 'update'):
        super().__init__({}, uses_grad=False, target = target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        prev = unpack_states(states, tensors, 'prev', cls=TensorList)
        mask = prev.mul_(tensors).gt_(0)
        prev.copy_(tensors)
        return mask


class SignConsistencyLRs(Transform):
    """Outputs per-weight learning rates based on consecutive sign consistency.

    The learning rate for a weight is multiplied by :code:`nplus` when two consecutive update signs are the same, otherwise it is multiplied by :code:`nplus`. The learning rates are bounded to be in :code:`(lb, ub)` range.

    Examples:

        GD scaled by consecutive gradient sign consistency

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.Mul(tz.m.SignConsistencyLRs()),
                tz.m.LR(1e-2)
            )

    """
    def __init__(
        self,
        nplus: float = 1.2,
        nminus: float = 0.5,
        lb: float | None = 1e-6,
        ub: float | None = 50,
        alpha: float = 1,
        target: Target = 'update'
    ):
        defaults = dict(nplus = nplus, nminus = nminus, alpha = alpha, lb = lb, ub = ub)
        super().__init__(defaults, uses_grad=False, target = target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        target = as_tensorlist(tensors)
        nplus, nminus, lb, ub = unpack_dicts(settings, 'nplus', 'nminus', 'lb', 'ub', cls=NumberList)
        prev, lrs = unpack_states(states, tensors, 'prev', 'lrs', cls=TensorList)

        if step == 0:
            lrs.set_(target.full_like([s['alpha'] for s in settings]))

        target = sign_consistency_lrs_(
            tensors = target,
            prev_ = prev,
            lrs_ = lrs,
            nplus = nplus,
            nminus = nminus,
            lb = lb,
            ub = ub,
            step = step,
        )
        return target.clone()
