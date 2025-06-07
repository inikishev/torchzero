from collections import deque
from collections.abc import Iterable
from operator import itemgetter
from typing import Literal

import torch

from ...core import Chainable, Module, Target, TensorwiseTransform, Transform, Var
from ...utils import Distributions, NumberList, TensorList, unpack_dicts, unpack_states


class Previous(TensorwiseTransform):
    """Maintains an update from n steps back, for example if n=1, returns previous update"""
    def __init__(self, n=1, target: Target = 'update'):
        defaults = dict(n=n)
        super().__init__(uses_grad=False, defaults=defaults, target=target)


    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, loss, state, settings):
        n = settings['n']

        if 'history' not in state:
            state['history'] = deque(maxlen=n+1)

        state['history'].append(tensor)

        return state['history'][0]


class LastDifference(Transform):
    """Difference between past two updates."""
    def __init__(self,target: Target = 'update'):
        super().__init__({}, uses_grad=False, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        prev = unpack_states(states, tensors, 'prev_target') # initialized to 0
        difference = torch._foreach_sub(tensors, prev)
        for p, c in zip(prev, tensors): p.set_(c)
        return difference

class LastGradDifference(Module):
    """Difference between past two grads."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def step(self, var):
        grad = var.get_grad()
        prev_grad = self.get_state(var.params, 'prev_grad') # initialized to 0
        difference = torch._foreach_sub(grad, prev_grad)
        for p, c in zip(prev_grad, grad): p.set_(c)
        var.update = list(difference)
        return var


class LastProduct(Transform):
    """Difference between past two updates."""
    def __init__(self,target: Target = 'update'):
        super().__init__({}, uses_grad=False, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        prev = unpack_states(states, tensors, 'prev', init=torch.ones_like) # initialized to 1 for prod
        prod = torch._foreach_mul(tensors, prev)
        for p, c in zip(prev, tensors): p.set_(c)
        return prod

class LastRatio(Transform):
    """Ratio between past two updates."""
    def __init__(self, numerator: Literal['cur', 'prev'] = 'cur', target: Target = 'update'):
        defaults = dict(numerator=numerator)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        prev = unpack_states(states, tensors, 'prev', init = torch.ones_like) # initialized to ones
        numerator = settings[0]['numerator']
        if numerator == 'cur': ratio = torch._foreach_div(tensors, prev)
        else: ratio = torch._foreach_div(prev, tensors)
        for p, c in zip(prev, tensors): p.set_(c)
        return ratio

class LastAbsoluteRatio(Transform):
    """Ratio between absolute values of past two updates."""
    def __init__(self, numerator: Literal['cur', 'prev'] = 'cur', eps:float=1e-8, target: Target = 'update'):
        defaults = dict(numerator=numerator, eps=eps)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        prev = unpack_states(states, tensors, 'prev', init = torch.ones_like) # initialized to ones
        numerator = settings[0]['numerator']
        eps = NumberList(s['eps'] for s in settings)

        torch._foreach_abs_(tensors)
        torch._foreach_clamp_min_(prev, eps)

        if numerator == 'cur': ratio = torch._foreach_div(tensors, prev)
        else: ratio = torch._foreach_div(prev, tensors)
        for p, c in zip(prev, tensors): p.set_(c)
        return ratio

class GradSign(Transform):
    """copy gradient sign to update."""
    def __init__(self, target: Target = 'update'):
        super().__init__({}, uses_grad=True, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        return [t.copysign_(g) for t,g in zip(tensors, grads)]

class UpdateSign(Transform):
    """use per-weight magnitudes from grad while using sign from update."""
    def __init__(self, target: Target = 'update'):
        super().__init__({}, uses_grad=True, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        return [g.copysign(t) for t,g in zip(tensors, grads)] # no in-place

class GraftToGrad(Transform):
    """use gradient norm and update direction."""
    def __init__(self, tensorwise:bool=False, ord:float=2, eps:float = 1e-6, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=True, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(settings[0])
        return TensorList(tensors).graft_(grads, tensorwise=tensorwise, ord=ord, eps=eps)

class GraftGradToUpdate(Transform):
    """use update norm and gradient direction."""
    def __init__(self, tensorwise:bool=False, ord:float=2, eps:float = 1e-6, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=True, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(settings[0])
        return TensorList(grads).graft(tensors, tensorwise=tensorwise, ord=ord, eps=eps)


class GraftToParams(Transform):
    """makes update norm be set to parameter norm, but norm won't go below eps"""
    def __init__(self, tensorwise:bool=False, ord:float=2, eps:float = 1e-4, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(settings[0])
        return TensorList(tensors).graft_(params, tensorwise=tensorwise, ord=ord, eps=eps)

class Relative(Transform):
    """multiplies update by absolute parameter values to make it relative to their magnitude, min_value is minimum value to avoid getting stuck at 0"""
    def __init__(self, min_value:float = 1e-4, target: Target = 'update'):
        defaults = dict(min_value=min_value)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        mul = TensorList(params).abs().clamp_([s['min_value'] for s in settings])
        torch._foreach_mul_(tensors, mul)
        return tensors

class FillLoss(Module):
    """makes tensors filled with loss value times alpha"""
    def __init__(self, alpha: float = 1, backward: bool = True):
        defaults = dict(alpha=alpha, backward=backward)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        alpha = self.get_settings(var.params, 'alpha')
        loss = var.get_loss(backward=self.settings[var.params[0]]['backward'])
        var.update = [torch.full_like(p, loss*a) for p,a in zip(var.params, alpha)]
        return var

class MulByLoss(Module):
    """multiplies update by loss times alpha"""
    def __init__(self, alpha: float = 1, min_value:float = 1e-8, backward: bool = True):
        defaults = dict(alpha=alpha, min_value=min_value, backward=backward)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        alpha, min_value = self.get_settings(var.params, 'alpha', 'min_value')
        loss = var.get_loss(backward=self.settings[var.params[0]]['backward'])
        mul = [max(loss*a, mv) for a,mv in zip(alpha, min_value)]
        torch._foreach_mul_(var.update, mul)
        return var

class DivByLoss(Module):
    """divides update by loss times alpha"""
    def __init__(self, alpha: float = 1, min_value:float = 1e-8, backward: bool = True):
        defaults = dict(alpha=alpha, min_value=min_value, backward=backward)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        alpha, min_value = self.get_settings(var.params, 'alpha', 'min_value')
        loss = var.get_loss(backward=self.settings[var.params[0]]['backward'])
        mul = [max(loss*a, mv) for a,mv in zip(alpha, min_value)]
        torch._foreach_div_(var.update, mul)
        return var



def _sequential_step(self: Module, var: Var, sequential: bool):
    params = var.params
    steps = self.settings[params[0]]['steps']

    if sequential: modules = self.get_children_sequence()
    else: modules = [self.children['module']] * steps

    if var.closure is None and len(modules) > 1: raise ValueError('Multistep and Sequential require closure')

    # store original params unless this is last module and can update params directly
    params_before_steps = None if (var.is_last and var.last_module_lrs is None) else [p.clone() for p in params]

    # first step - pass var as usual
    var = modules[0].step(var)
    new_var = var

    # subsequent steps - update parameters and create new var
    if len(modules) > 1:
        for m in modules[1:]:

            # update params
            if (not new_var.skip_update):
                if new_var.last_module_lrs is not None:
                    torch._foreach_mul_(new_var.get_update(), new_var.last_module_lrs)

                torch._foreach_sub_(params, new_var.get_update())

            # create new var since we are at a new point, that means grad, update and loss will be None
            new_var = Var(params=new_var.params, closure=new_var.closure,
                            model=new_var.model, current_step=new_var.current_step + 1)

            # step
            new_var = m.step(new_var)

        # final parameter update
        if (not new_var.skip_update):
            if new_var.last_module_lrs is not None:
                torch._foreach_mul_(new_var.get_update(), new_var.last_module_lrs)

            torch._foreach_sub_(params, new_var.get_update())

    # if last module, update is applied so return new var
    if params_before_steps is None:
        new_var.stop = True
        new_var.skip_update = True
        return new_var

    # otherwise use parameter difference as update
    var.update = list(torch._foreach_sub(params_before_steps, params))
    for p, bef in zip(params, params_before_steps):
        p.set_(bef) # pyright:ignore[reportArgumentType]
    return var

class Multistep(Module):
    def __init__(self, module: Chainable, steps: int):
        defaults = dict(steps=steps)
        super().__init__(defaults)
        self.set_child('module', module)

    @torch.no_grad
    def step(self, var):
        return _sequential_step(self, var, sequential=False)

class Sequential(Module):
    def __init__(self, modules: Iterable[Chainable], steps: int):
        defaults = dict(steps=steps)
        super().__init__(defaults)
        self.set_children_sequence(modules)

    @torch.no_grad
    def step(self, var):
        return _sequential_step(self, var, sequential=True)


class GradientAccumulation(Module):
    """gradient accumulation"""
    def __init__(self, modules: Chainable, n: int, mean=True, stop=True):
        defaults = dict(n=n, mean=mean, stop=stop)
        super().__init__(defaults)
        self.set_child('modules', modules)


    @torch.no_grad
    def step(self, var):
        accumulator = self.get_state(var.params, 'accumulator')
        settings = self.settings[var.params[0]]
        n = settings['n']; mean = settings['mean']; stop = settings['stop']
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        # add update to accumulator
        torch._foreach_add_(accumulator, var.get_update())

        # step with accumulated updates
        if step % n == 0:
            if mean:
                torch._foreach_div_(accumulator, n)

            var.update = [a.clone() for a in accumulator]
            var = self.children['modules'].step(var)

            # zero accumulator
            torch._foreach_zero_(accumulator)

        else:
            # prevent update
            if stop:
                var.stop=True
                var.skip_update=True

        return var


class Dropout(Transform):
    def __init__(self, p: float = 0.5, graft: bool=False, target: Target = 'update'):
        defaults = dict(p=p, graft=graft)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        p = NumberList(s['p'] for s in settings)
        graft = settings[0]['graft']

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
    def step(self, var):
        closure = var.closure
        if closure is None: raise RuntimeError('WeightDropout requires closure')
        params = TensorList(var.params)
        p = NumberList(self.settings[p]['p'] for p in params)
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

        var.closure = dropout_closure
        return var

class NoiseSign(Transform):
    """uses random vector with update sign"""
    def __init__(self, distribution:Distributions = 'normal', alpha = 1):
        defaults = dict(distribution=distribution, alpha=alpha)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        alpha = [s['alpha'] for s in settings]
        distribution = self.settings[params[0]]['distribution']
        return TensorList(tensors).sample_like(alpha, distribution).copysign_(tensors)


class NegateOnLossIncrease(Module):
    def __init__(self, backtrack=True):
        defaults = dict(backtrack=backtrack)
        super().__init__(defaults=defaults)

    @torch.no_grad
    def step(self, var):
        closure = var.closure
        if closure is None: raise RuntimeError('NegateOnLossIncrease requires closure')
        backtrack = self.settings[var.params[0]]['backtrack']

        update = var.get_update()
        f_0 = var.get_loss(backward=False)

        torch._foreach_sub_(var.params, update)
        f_1 = closure(False)

        if f_1 <= f_0:
            if var.is_last and var.last_module_lrs is None:
                var.stop = True
                var.skip_update = True
                return var

            torch._foreach_add_(var.params, update)
            return var

        torch._foreach_add_(var.params, update)
        if backtrack:
            torch._foreach_neg_(var.update)
        else:
            torch._foreach_zero_(var.update)
        return var