from collections import abc

import torch

from ...tensorlist import TensorList, where, Distributions
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
        if vars.ascent is not None: raise ValueError("Minibatch Rprop must be the first module.")
        params = self.get_params()

        nplus, nminus, lb, ub = self.get_group_keys('nplus', 'nminus', 'lb', 'ub')
        allowed, magnitudes = self.get_state_keys(
            'allowed', 'magnitudes',
            inits = [_bool_ones_like, torch.zeros_like],
            params=params
        )

        g1_sign = vars.maybe_compute_grad_(params).sign() # no inplace to not modify grads
        # initialize on 1st iteration
        if self.current_step == 0:
            magnitudes.fill_(self.get_group_key('alpha')).clamp_(lb, ub)
            # ascent = magnitudes * g1_sign
            # self.current_step += 1
            # return ascent

        # first step
        ascent = g1_sign.mul_(magnitudes).mul_(allowed)
        params -= ascent
        with torch.enable_grad(): vars.fx0_approx = vars.closure()
        f0 = vars.fx0; f1 = vars.fx0_approx
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
            vars.fx0 = vars.fx0_approx
            vars.grad = params.ensure_grad_().grad
            sign = vars.grad.sign()

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
            vars.ascent = ascent2
            return self._update_params_or_step_with_next(vars, params)

        if self.next_mode == 'add':
            # undo 1st step
            params += ascent
            vars.ascent = ascent + ascent2
            return self._update_params_or_step_with_next(vars, params)

        if self.next_mode == 'undo':
            params += ascent
            vars.ascent = ascent2
            return self._update_params_or_step_with_next(vars, params)

        raise ValueError(f'invalid next_mode: {self.next_mode}')



class GradMin(OptimizerModule):
    """
    for experiments, unlikely to work well on most problems.

    explanation: calculate grads wrt sum of grads + loss.
    """
    def __init__(self, loss_term: float = 1, square=False, maximize_grad = False, create_graph = False):
        super().__init__(dict(loss_term=loss_term))
        self.square = square
        self.maximize_grad = maximize_grad
        self.create_graph = create_graph

    @torch.no_grad
    def step(self, vars):
        if vars.closure is None: raise ValueError()
        if vars.ascent is not None:
            raise ValueError("GradMin doesn't accept ascent_direction")

        params = self.get_params()
        loss_term = self.get_group_key('loss_term')

        self.zero_grad()
        with torch.enable_grad():
            vars.fx0 = vars.closure(False)
            grads = jacobian([vars.fx0], params, create_graph=True, batched=False) # type:ignore
            grads = TensorList(grads).squeeze_(0)
            if self.square:
                grads = grads ** 2
            else:
                grads = grads.abs()

            if self.maximize_grad: grads: TensorList = grads - (vars.fx0 * loss_term) # type:ignore
            else: grads = grads + (vars.fx0 * loss_term)
            grad_mean = torch.sum(torch.stack(grads.sum())) / grads.total_numel()

            if self.create_graph: grad_mean.backward(create_graph=True)
            else: grad_mean.backward(retain_graph=False)

        if self.maximize_grad: vars.grad = params.ensure_grad_().grad.neg_()
        else: vars.grad = params.ensure_grad_().grad

        vars.maybe_use_grad_(params)
        return self._update_params_or_step_with_next(vars)


class HVPDiagNewton(OptimizerModule):
    """
    for experiments, unlikely to work well on most problems.

    explanation: may or may not approximate newton step if hessian is diagonal with 2 backward passes. Probably not.
    """
    def __init__(self, eps=1e-3):
        super().__init__(dict(eps=eps))

    @torch.no_grad
    def step(self, vars):
        if vars.closure is None: raise ValueError()
        if vars.ascent is not None:
            raise ValueError("HVPDiagNewton doesn't accept ascent_direction")

        params = self.get_params()
        eps = self.get_group_key('eps')
        grad_fx0 = vars.maybe_compute_grad_(params).clone()
        vars.grad = grad_fx0 # set state grad to the cloned version, since it will be overwritten

        params += grad_fx0 * eps
        with torch.enable_grad(): _ = vars.closure()

        params -= grad_fx0 * eps

        newton = grad_fx0 * ((grad_fx0 * eps) / (params.grad - grad_fx0))
        newton.nan_to_num_(0,0,0)

        vars.ascent = newton
        return self._update_params_or_step_with_next(vars)



class ReduceOutwardLR(OptimizerModule):
    """
    When update sign matches weight sign, the learning rate for that weight is multiplied by `mul`.

    This means updates that move weights towards zero have higher learning rates.
    """
    def __init__(self, mul = 0.5, use_grad=False, invert=False):
        defaults = dict(mul = mul)
        super().__init__(defaults)

        self.use_grad = use_grad
        self.invert = invert

    @torch.no_grad
    def _update(self, vars, ascent):
        params = self.get_params()
        mul = self.get_group_key('mul')

        if self.use_grad: cur = vars.maybe_compute_grad_(params)
        else: cur = ascent

        # mask of weights where sign matches with update sign (minus ascent sign), multiplied by `mul`.
        if self.invert: mask = (params * cur) > 0
        else: mask = (params * cur) < 0
        ascent.masked_set_(mask, ascent*mul)

        return ascent

class NoiseSign(OptimizerModule):
    """uses random vector with ascent sign"""
    def __init__(self, distribution:Distributions = 'normal', alpha = 1):
        super().__init__({})
        self.alpha = alpha
        self.distribution:Distributions = distribution


    def _update(self, vars, ascent):
        return ascent.sample_like(self.alpha, self.distribution).copysign_(ascent)

class ParamSign(OptimizerModule):
    """uses params with ascent sign"""
    def __init__(self):
        super().__init__({})


    def _update(self, vars, ascent):
        params = self.get_params()

        return params.copysign(ascent)

class NegParamSign(OptimizerModule):
    """uses max(params_abs) - params_abs with ascent sign"""
    def __init__(self):
        super().__init__({})


    def _update(self, vars, ascent):
        neg_params = self.get_params().abs()
        max = neg_params.total_max()
        neg_params = neg_params.neg_().add(max)
        return neg_params.copysign_(ascent)

class InvParamSign(OptimizerModule):
    """uses 1/(params_abs+eps) with ascent sign"""
    def __init__(self, eps=1e-2):
        super().__init__({})
        self.eps = eps


    def _update(self, vars, ascent):
        inv_params = self.get_params().abs().add_(self.eps).reciprocal_()
        return inv_params.copysign(ascent)


class ParamWhereConsistentSign(OptimizerModule):
    """where ascent and param signs are the same, it sets ascent to param value"""
    def __init__(self, eps=1e-2):
        super().__init__({})
        self.eps = eps


    def _update(self, vars, ascent):
        params = self.get_params()
        same_sign = params.sign() == ascent.sign()
        ascent.masked_set_(same_sign, params)

        return ascent