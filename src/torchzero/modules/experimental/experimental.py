from collections import abc

import torch

from ...tensorlist import TensorList, where
from ...core import OptimizerModule
from ...utils.derivatives import jacobian

def _bool_ones_like(x):
    return torch.ones_like(x, dtype=torch.bool)


class MinibatchRprop(OptimizerModule):
    """
    for experiments, unlikely to work well on most problems.

    explanation: does 2 steps per batch, applies rprop rule on the second step.
    """
    def __init__(
        self,
        lr: float = 1,
        nplus: float = 1.2,
        nminus: float = 0.5,
        lb: float | None = 1e-6,
        ub: float | None = 50,
        backtrack=True,
        next_mode = 'continue',
        increase_mul = 0.5,
    ):
        defaults = dict(nplus = nplus, nminus = nminus, lr = lr, lb = lb, ub = ub, increase_mul=increase_mul)
        super().__init__(defaults)
        self.current_step = 0
        self.backtrack = backtrack

        self.next_mode = next_mode

    @torch.no_grad
    def step(self, state):
        if state.closure is None: raise ValueError("Minibatch Rprop requires closure")
        if state.ascent is not None: raise ValueError("Minibatch Rprop must be the first module.")
        params = self.get_params()

        nplus, nminus, lb, ub = self.get_group_keys(['nplus', 'nminus', 'lb', 'ub'])
        allowed, magnitudes = self.get_state_keys(
            ['allowed', 'magnitudes'],
            inits = [_bool_ones_like, torch.zeros_like],
            params=params
        )

        g1_sign = state.maybe_compute_grad_(params).sign() # no inplace to not modify grads
        # initialize on 1st iteration
        if self.current_step == 0:
            magnitudes.fill_(self.defaults['lr']).clamp_(lb, ub)
            # ascent = magnitudes * g1_sign
            # self.current_step += 1
            # return ascent

        # first step
        ascent = g1_sign.mul_(magnitudes).mul_(allowed)
        params -= ascent
        with torch.enable_grad(): state.fx0_approx = state.closure()
        f0 = state.fx0; f1 = state.fx0_approx
        assert f0 is not None and f1 is not None

        # if loss increased, reduce all lrs and undo the update
        if f1 > f0:
            increase_mul = self.get_group_key('increase_mul')
            magnitudes.mul_(increase_mul).clamp_(lb, ub)
            params += ascent
            self.current_step += 1
            return f0

        # on `continue` we move to params after 1st update
        # therefore state must be updated to have all attributes after 1st update
        if self.next_mode == 'continue':
            state.fx0 = state.fx0_approx
            state.grad = params.ensure_grad_().grad
            sign = state.grad.sign()

        else:
            sign = params.ensure_grad_().grad.sign_() # can use in-place as this is not fx0 grad

        # compare 1st and 2nd gradients via rprop rule
        prev = ascent
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
            ascent2 = sign.mul_(magnitudes)
            ascent2.masked_set_(sign_changed, prev.neg_())
        else:
            ascent2 = sign.mul_(magnitudes * ~sign_changed)

        # update allowed to only have weights where last update wasn't reverted
        allowed.set_(sign_same | zeroes)

        self.current_step += 1

        # update params or step
        if self.next_mode == 'continue' or (self.next_mode == 'add' and self.next_module is None):
            state.ascent = ascent2
            return self._update_params_or_step_with_next(state, params)

        elif self.next_mode == 'add':
            # undo 1st step
            params += ascent
            state.ascent = ascent + ascent2
            return self._update_params_or_step_with_next(state, params)

        elif self.next_mode == 'undo':
            params += ascent
            state.ascent = ascent2
            return self._update_params_or_step_with_next(state, params)

        else:
            raise ValueError(f'invalid next_mode: {self.next_mode}')



class GradMin(OptimizerModule):
    """
    for experiments, unlikely to work well on most problems.

    explanation: calculate grads wrt sum of grads + loss.
    """
    def __init__(self, loss_term: float = 1, square=False, maximize_grad = False):
        super().__init__(dict(add_loss=loss_term))
        self.square = square
        self.maximize_grad = maximize_grad

    @torch.no_grad
    def step(self, state):
        if state.closure is None: raise ValueError()
        if state.ascent is not None:
            raise ValueError("GradMin doesn't accept ascent_direction")

        params = self.get_params()
        add_loss = self.get_group_key('add_loss')

        self.zero_grad()
        with torch.enable_grad():
            state.fx0 = state.closure(False)
            grads = jacobian([state.fx0], params, create_graph=True, batched=False) # type:ignore
            grads = TensorList(grads).squeeze_(0)
            if self.square:
                grads = grads ** 2
            else:
                grads = grads.abs()

            if self.maximize_grad: grads: TensorList = grads - (state.fx0 * add_loss) # type:ignore
            else: grads = grads + (state.fx0 * add_loss)
            grad_mean = torch.sum(torch.stack(grads.sum())) / grads.total_numel()
            grad_mean.backward(retain_graph=False)

        if self.maximize_grad: state.grad = params.ensure_grad_().grad.neg_()
        else: state.grad = params.ensure_grad_().grad

        state.maybe_use_grad_(params)
        return self._update_params_or_step_with_next(state)


class HVPDiagNewton(OptimizerModule):
    """
    for experiments, unlikely to work well on most problems.

    explanation: should approximate newton step if hessian is diagonal.
    """
    def __init__(self, eps=1e-3):
        super().__init__(dict(eps=eps))

    @torch.no_grad
    def step(self, state):
        if state.closure is None: raise ValueError()
        if state.ascent is not None:
            raise ValueError("HVPDiagNewton doesn't accept ascent_direction")

        params = self.get_params()
        eps = self.get_group_key('eps')
        grad_fx0 = state.maybe_compute_grad_(params).clone()
        state.grad = grad_fx0 # set state grad to the cloned version, since it will be overwritten

        params += grad_fx0 * eps
        with torch.enable_grad(): _ = state.closure()

        params -= grad_fx0 * eps

        newton = grad_fx0 * ((grad_fx0 * eps) / (params.grad - grad_fx0))
        newton.nan_to_num_(0,0,0)

        state.ascent = newton
        return self._update_params_or_step_with_next(state)


def _reset_stats_hook(optimizer, state):
    for module in optimizer.modules:
        module: OptimizerModule
        module.reset_stats()

class CyclicSWA(OptimizerModule):
    """I remember reading about this but I have no idea if I actually red it or if I dreamed it up. I am not able to find the paper.

    This is just periodic SWA with cyclic learning rate. So it samples the weights, increases lr to `peak_lr`, samples the weights again,
    decreases lr back to `init_lr`, and samples the weights last time. Then model weights are replaced with the average of the three sampled weights,
    and next cycle starts.

    It is easier to tune than PeriodicSWA and seems to work better too.
    """
    def __init__(self, cswa_start: int, cycle_length: int, steps_between: int, init_lr: float = 0, peak_lr: float = 1):

        super().__init__({})
        self.cswa_start = cswa_start
        self.cycle_length = cycle_length
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.steps_between = steps_between

        self.cur = 0
        self.cycle_cur = 0
        self.n_models = 0

        self.cur_lr = self.init_lr

    def step(self, state):
        params = self.get_params()

        # start first period after `cswa_start` steps
        if self.cur >= self.cswa_start:

            ascent = state.maybe_use_grad_(params)

            # determine the lr
            point = self.cycle_cur / self.cycle_length
            if point < 0.5:
                p2 = point*2
                lr = self.init_lr * (1-p2) + self.peak_lr * p2
            else:
                p2 = (1 - point)*2
                lr = self.init_lr * (1-p2) + self.peak_lr * p2

            ascent *= lr
            ret = self._update_params_or_step_with_next(state, params)

            if self.cycle_cur in (0, self.cycle_length, self.cycle_length // 2):
                swa = self.get_state_key('swa')
                swa.mul_(self.n_models).add_(params).div_(self.n_models + 1)
                self.n_models += 1

                if self.cycle_cur == self.cycle_length:
                    assert self.n_models == 3, self.n_models
                    self.n_models = 0
                    self.cycle_cur = -1

                    params.set_(swa)
                    state.add_post_step_hook(_reset_stats_hook)

            self.cycle_cur += 1

        else:
            ret = self._update_params_or_step_with_next(state, params)

        self.cur += 1

        return ret