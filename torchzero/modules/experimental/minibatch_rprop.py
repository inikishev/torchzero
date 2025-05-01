import torch

from ...core import Module
from ...utils import NumberList, TensorList

def _true_like(x):
    return torch.ones_like(x, dtype=torch.bool)

class MinibatchRprop(Module):
    """
    for experiments, unlikely to work well on most problems.

    explanation: does 2 steps per batch, applies rprop rule on the second step.
    """
    def __init__(
        self,
        nplus: float = 1.2,
        nminus: float = 0.5,
        lb: float | None = 1e-6,
        ub: float | None = 50,
        backtrack=True,
        next_mode = 'continue',
        increase_mul = 0.5,
        alpha: float = 1,
    ):
        defaults = dict(nplus = nplus, nminus = nminus, alpha = alpha, lb = lb, ub = ub, increase_mul=increase_mul)
        super().__init__(defaults)
        self.current_step = 0
        self.backtrack = backtrack
        self.next_mode = next_mode

    @torch.no_grad
    def step(self, vars):
        if vars.closure is None: raise ValueError("Minibatch Rprop requires closure")
        # if vars.update is not None: raise ValueError("Minibatch Rprop must be the first module.")
        params = TensorList(vars.params)

        nplus, nminus, lb, ub = self.get_settings('nplus', 'nminus', 'lb', 'ub', params=params, cls=NumberList)
        allowed, magnitudes = self.get_state('allowed', 'magnitudes', params=params, init = [_true_like, torch.zeros_like], cls=TensorList)

        g1_sign = TensorList(vars.get_grad()).sign() # no inplace to not modify grads
        # initialize on 1st iteration
        if self.current_step == 0:
            magnitudes.fill_(self.get_settings('alpha', params=params)).clamp_(lb, ub)
            # ascent = magnitudes * g1_sign
            # self.current_step += 1
            # return ascent

        # first step
        update = g1_sign.mul_(magnitudes).mul_(allowed)
        params -= update
        with torch.enable_grad(): vars.loss_approx = vars.closure()
        f_0 = vars.loss; f_1 = vars.loss_approx
        assert f_0 is not None and f_1 is not None

        # if loss increased, reduce all lrs and undo the update
        if f_1 > f_0:
            increase_mul = self.get_settings('increase_mul', params=params)
            magnitudes.mul_(increase_mul).clamp_(lb, ub)
            params += update
            self.current_step += 1
            vars.update = None
            return vars

        # on `continue` we move to params after 1st update
        # therefore state must be updated to have all attributes after 1st update
        if self.next_mode == 'continue':
            vars.loss = vars.loss_approx
            vars.grad = params.ensure_grad_().grad
            sign = vars.grad.sign()

        else:
            sign = params.ensure_grad_().grad.sign_() # can use in-place as this is not fx0 grad

        # compare 1st and 2nd gradients via rprop rule
        prev = update
        mul = sign * prev # prev is already multiuplied by `allowed`

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
            new_update = sign.mul_(magnitudes)
            new_update.masked_set_(sign_changed, prev.neg_())
        else:
            new_update = sign.mul_(magnitudes * ~sign_changed)

        # update allowed to only have weights where last update wasn't reverted
        allowed.set_(sign_same | zeroes)

        self.current_step += 1

        # update params or step
        if self.next_mode == 'continue':
            vars.update = new_update
            return vars

        if self.next_mode == 'add':
            # undo 1st step
            params += update
            vars.update = update + new_update
            return vars

        if self.next_mode == 'undo':
            params += update
            vars.update = new_update
            return vars

        raise ValueError(f'invalid next_mode: {self.next_mode}')
