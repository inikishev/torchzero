from collections import deque
from collections.abc import Iterable
from operator import itemgetter
from typing import Literal

import torch

from ...core import Chainable, Module, TensorwiseTransform, Target, Transform, Vars
from ...utils import Distributions, NumberList, TensorList


class Previous(TensorwiseTransform):
    """Maintains an update from n steps back, for example if n=1, returns previous update"""
    def __init__(self, n=1, target: Target = 'update'):
        defaults = dict(n=n)
        super().__init__(uses_grad=False, defaults=defaults, target=target)


    @torch.no_grad
    def transform(self, tensor, param, grad, vars):
        n = self.settings[param]['n']
        state = self.state[param]

        if 'history' not in state:
            state['history'] = deque(maxlen=n+1)

        state['history'].append(tensor)

        return state['history'][0]


class LastDifference(Transform):
    """Difference between past two updates."""
    def __init__(self,target: Target = 'update'):
        super().__init__({}, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        prev_target = self.get_state('prev_target', params=params) # initialized to 0
        difference = torch._foreach_sub(tensors, prev_target)
        for p, c in zip(prev_target, tensors): p.set_(c)
        return difference

class LastGradDifference(Module):
    """Difference between past two grads."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def step(self, vars):
        grad = vars.get_grad()
        prev_grad = self.get_state('prev_grad', params=vars.params) # initialized to 0
        difference = torch._foreach_sub(grad, prev_grad)
        for p, c in zip(prev_grad, grad): p.set_(c)
        vars.update = list(difference)
        return vars


class LastProduct(Transform):
    """Difference between past two updates."""
    def __init__(self,target: Target = 'update'):
        super().__init__({}, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        prev_target = self.get_state('prev_target', params=params, init=torch.ones_like) # initialized to 1 for prod
        prod = torch._foreach_mul(tensors, prev_target)
        for p, c in zip(prev_target, tensors): p.set_(c)
        return prod

class LastRatio(Transform):
    """Ratio between past two updates."""
    def __init__(self, numerator: Literal['cur', 'prev'] = 'cur', target: Target = 'update'):
        defaults = dict(numerator=numerator)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        prev_target = self.get_state('prev_target', params=params, init = torch.ones_like) # initialized to ones
        numerator = self.settings[params[0]]['numerator']
        if numerator == 'cur': ratio = torch._foreach_div(tensors, prev_target)
        else: ratio = torch._foreach_div(prev_target, tensors)
        for p, c in zip(prev_target, tensors): p.set_(c)
        return ratio

class LastAbsoluteRatio(Transform):
    """Ratio between absolute values of past two updates."""
    def __init__(self, numerator: Literal['cur', 'prev'] = 'cur', eps:float=1e-8, target: Target = 'update'):
        defaults = dict(numerator=numerator, eps=eps)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        prev_target = self.get_state('prev_target', params=params, init = torch.ones_like) # initialized to 0
        numerator = self.settings[params[0]]['numerator']
        eps = self.get_settings('eps', params=params, cls = NumberList)

        torch._foreach_abs_(tensors)
        torch._foreach_clamp_min_(prev_target, eps)

        if numerator == 'cur': ratio = torch._foreach_div(tensors, prev_target)
        else: ratio = torch._foreach_div(prev_target, tensors)
        for p, c in zip(prev_target, tensors): p.set_(c)
        return ratio

class GradSign(Transform):
    """copy gradient sign to update."""
    def __init__(self, target: Target = 'update'):
        super().__init__({}, uses_grad=True, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        assert grads is not None
        return [t.copysign_(g) for t,g in zip(tensors, grads)]

class UpdateSign(Transform):
    """use per-weight magnitudes from grad while using sign from update."""
    def __init__(self, target: Target = 'update'):
        super().__init__({}, uses_grad=True, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        assert grads is not None
        return [g.copysign(t) for t,g in zip(tensors, grads)] # no in-place

class GraftToGrad(Transform):
    """use gradient norm and update direction."""
    def __init__(self, tensorwise:bool=False, ord:float=2, eps:float = 1e-6, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=True, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        assert grads is not None
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(self.settings[params[0]])
        return TensorList(tensors).graft_(grads, tensorwise=tensorwise, ord=ord, eps=eps)

class GraftGradToUpdate(Transform):
    """use update norm and gradient direction."""
    def __init__(self, tensorwise:bool=False, ord:float=2, eps:float = 1e-6, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=True, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        assert grads is not None
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(self.settings[params[0]])
        return TensorList(grads).graft(tensors, tensorwise=tensorwise, ord=ord, eps=eps)


class GraftToParams(Transform):
    """makes update norm be set to parameter norm, but norm won't go below eps"""
    def __init__(self, tensorwise:bool=False, ord:float=2, eps:float = 1e-4, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(self.settings[params[0]])
        return TensorList(tensors).graft_(params, tensorwise=tensorwise, ord=ord, eps=eps)

class Relative(Transform):
    """multiplies update by absolute parameter values to make it relative to their magnitude, min_value is minimum value to avoid getting stuck at 0"""
    def __init__(self, min_value:float = 1e-4, target: Target = 'update'):
        defaults = dict(min_value=min_value)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        mul = TensorList(params).abs().clamp_(self.get_settings('min_value', params=params))
        torch._foreach_mul_(tensors, mul)
        return tensors

class FillLoss(Module):
    """makes tensors filled with loss value times alpha"""
    def __init__(self, alpha: float = 1, backward: bool = True):
        defaults = dict(alpha=alpha, backward=backward)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, vars):
        alpha = self.get_settings('alpha', params=vars.params)
        loss = vars.get_loss(backward=self.settings[vars.params[0]]['backward'])
        vars.update = [torch.full_like(p, loss*a) for p,a in zip(vars.params, alpha)]
        return vars

class MulByLoss(Transform):
    """multiplies update by loss times alpha"""
    def __init__(self, alpha: float = 1, min_value:float = 1e-8, backward: bool = True, target: Target = 'update'):
        defaults = dict(alpha=alpha, min_value=min_value, backward=backward)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars): #vars used for loss
        alpha, min_value = self.get_settings('alpha', 'min_value', params=params)
        loss = vars.get_loss(backward=self.settings[params[0]]['backward'])
        mul = [max(loss*a, mv) for a,mv in zip(alpha, min_value)]
        torch._foreach_mul_(tensors, mul)
        return tensors

class DivByLoss(Transform):
    """divides update by loss times alpha"""
    def __init__(self, alpha: float = 1, min_value:float = 1e-8, backward: bool = True, target: Target = 'update'):
        defaults = dict(alpha=alpha, min_value=min_value, backward=backward)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars): #vars used for loss
        alpha, min_value = self.get_settings('alpha', 'min_value', params=params)
        loss = vars.get_loss(backward=self.settings[params[0]]['backward'])
        mul = [max(loss*a, mv) for a,mv in zip(alpha, min_value)]
        torch._foreach_div_(tensors, mul)
        return tensors



def _sequential_step(self: Module, vars: Vars, sequential: bool):
    params = vars.params
    steps = self.settings[params[0]]['steps']

    if sequential: modules = self.get_children_sequence()
    else: modules = [self.children['module']] * steps

    if vars.closure is None and len(modules) > 1: raise ValueError('Multistep and Sequential require closure')

    # store original params unless this is last module and can update params directly
    params_before_steps = None if (vars.is_last and vars.last_module_lrs is None) else [p.clone() for p in params]

    # first step - pass vars as usual
    vars = modules[0].step(vars)
    new_vars = vars

    # subsequent steps - update parameters and create new vars
    if len(modules) > 1:
        for m in modules[1:]:

            # update params
            if (not new_vars.skip_update):
                if new_vars.last_module_lrs is not None:
                    torch._foreach_mul_(new_vars.get_update(), new_vars.last_module_lrs)

                torch._foreach_sub_(params, new_vars.get_update())

            # create new vars since we are at a new point, that means grad, update and loss will be None
            new_vars = Vars(params=new_vars.params, closure=new_vars.closure,
                            model=new_vars.model, current_step=new_vars.current_step + 1)

            # step
            new_vars = m.step(new_vars)

        # final parameter update
        if (not new_vars.skip_update):
            if new_vars.last_module_lrs is not None:
                torch._foreach_mul_(new_vars.get_update(), new_vars.last_module_lrs)

            torch._foreach_sub_(params, new_vars.get_update())

    # if last module, update is applied so return new vars
    if params_before_steps is None:
        new_vars.stop = True
        new_vars.skip_update = True
        return new_vars

    # otherwise use parameter difference as update
    vars.update = list(torch._foreach_sub(params_before_steps, params))
    for p, bef in zip(params, params_before_steps):
        p.set_(bef) # pyright:ignore[reportArgumentType]
    return vars

class Multistep(Module):
    def __init__(self, module: Chainable, steps: int):
        defaults = dict(steps=steps)
        super().__init__(defaults)
        self.set_child('module', module)

    @torch.no_grad
    def step(self, vars):
        return _sequential_step(self, vars, sequential=False)

class Sequential(Module):
    def __init__(self, modules: Iterable[Chainable], steps: int):
        defaults = dict(steps=steps)
        super().__init__(defaults)
        self.set_children_sequence(modules)

    @torch.no_grad
    def step(self, vars):
        return _sequential_step(self, vars, sequential=True)


class GradientAccumulation(Module):
    """gradient accumulation"""
    def __init__(self, modules: Chainable, n: int, mean=True, stop=True):
        defaults = dict(n=n, mean=mean, stop=stop)
        super().__init__(defaults)
        self.set_child('modules', modules)


    @torch.no_grad
    def step(self, vars):
        accumulator = self.get_state('accumulator', params=vars.params)
        settings = self.settings[vars.params[0]]
        n = settings['n']; mean = settings['mean']; stop = settings['stop']
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        # add update to accumulator
        torch._foreach_add_(accumulator, vars.get_update())

        # step with accumulated updates
        if step % n == 0:
            if mean:
                torch._foreach_div_(accumulator, n)

            vars.update = [a.clone() for a in accumulator]
            vars = self.children['modules'].step(vars)

            # zero accumulator
            torch._foreach_zero_(accumulator)

        else:
            # prevent update
            if stop:
                vars.stop=True
                vars.skip_update=True

        return vars


class Dropout(Transform):
    def __init__(self, p: float = 0.5, graft: bool=False, target: Target = 'update'):
        defaults = dict(p=p, graft=graft)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        tensors = TensorList(tensors)
        p = self.get_settings('p', params=params, cls=NumberList)
        graft = self.settings[params[0]]['graft']

        if graft:
            target_norm = tensors.global_vector_norm()
            tensors.mul_(tensors.rademacher_like(1-p).add_(1).div_(2))
            return tensors.mul_(target_norm / tensors.global_vector_norm()) # graft

        return tensors.mul_(tensors.rademacher_like(1-p).add_(1).div_(2))

class WeightDropout(Module):
    """Applies dropout directly to weights."""
    def __init__(self, p: float = 0.5, graft: bool = True):
        defaults = dict(p=p, graft=graft)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, vars):
        closure = vars.closure
        if closure is None: raise RuntimeError('WeightDropout requires closure')
        params = TensorList(vars.params)
        p = self.get_settings('p', params=params)
        mask = params.rademacher_like(p).add_(1).div_(2).as_bool()

        @torch.no_grad
        def dropout_closure(backward=True):
            orig_params = params.clone()
            params.mul_(mask)
            if backward:
                with torch.enable_grad(): loss = closure()
            else:
                loss = closure(False)
            params.copy_(orig_params)
            return loss

        vars.closure = dropout_closure
        return vars

class NoiseSign(Transform):
    """uses random vector with update sign"""
    def __init__(self, distribution:Distributions = 'normal', alpha = 1):
        defaults = dict(distribution=distribution, alpha=alpha)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        alpha = self.get_settings('alpha', params=params)
        distribution = self.settings[params[0]]['distribution']
        return TensorList(tensors).sample_like(alpha, distribution).copysign_(tensors)


class NegateOnLossIncrease(Module):
    def __init__(self, backtrack=True):
        defaults = dict(backtrack=backtrack)
        super().__init__(defaults=defaults)

    @torch.no_grad
    def step(self, vars):
        closure = vars.closure
        if closure is None: raise RuntimeError('NegateOnLossIncrease requires closure')
        backtrack = self.settings[vars.params[0]]['backtrack']

        update = vars.get_update()
        f_0 = vars.get_loss(backward=False)

        torch._foreach_sub_(vars.params, update)
        f_1 = closure(False)

        if f_1 <= f_0:
            if vars.is_last and vars.last_module_lrs is None:
                vars.stop = True
                vars.skip_update = True
                return vars

            torch._foreach_add_(vars.params, update)
            return vars

        torch._foreach_add_(vars.params, update)
        if backtrack:
            torch._foreach_neg_(vars.update)
        else:
            torch._foreach_zero_(vars.update)
        return vars