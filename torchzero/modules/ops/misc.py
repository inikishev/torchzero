from collections.abc import Iterable
from collections import deque
from operator import itemgetter

import torch

from ...core import ParameterwiseTransform, Target, Transform, Module, Chainable, Vars
from ...utils import TensorList

class Previous(ParameterwiseTransform):
    """Maintains an update from n steps back, for example if n=1, returns previous update"""
    def __init__(self, n=1, target: Target = 'update'):
        defaults = dict(n=n)
        super().__init__(requires_grad=False, defaults=defaults, target=target)


    @torch.no_grad
    def transform(self, target, param, grad, vars):
        n = self.settings[param]['n']
        state = self.state[param]

        if 'history' not in state:
            state['history'] = deque(maxlen=n+1)

        state['history'].append(target)

        return state['history'][0]


class LastDifference(Transform):
    """Difference between past two updates."""
    def __init__(self):
        super().__init__()

    @torch.no_grad
    def transform(self, target, vars):
        prev_target = self.get_state('prev_target', params=vars) # initialized to 0
        difference = torch._foreach_sub(target, prev_target)
        for p, c in zip(prev_target, target): p.set_(c)
        return difference

class LastGradDifference(Transform):
    """Difference between past two grads."""
    def __init__(self,target: Target = 'update'):
        super().__init__(target=target)

    @torch.no_grad
    def transform(self, target, vars):
        grad = vars.get_grad()
        prev_grad = self.get_state('prev_grad', params=vars) # initialized to 0
        difference = torch._foreach_sub(grad, prev_grad)
        for p, c in zip(prev_grad, grad): p.set_(c)
        return difference


class LastProduct(Transform):
    """Product of past two updates."""
    def __init__(self):
        super().__init__()

    @torch.no_grad
    def transform(self, target, vars):
        prev_target = self.get_state('prev_target', params=vars) # initialized to 0
        prod = torch._foreach_mul(target, prev_target)
        for p, c in zip(prev_target, target): p.set_(c)
        return prod

class GradSign(Transform):
    """copy gradient sign to update."""
    def __init__(self, target: Target = 'update'):
        super().__init__(target=target)

    @torch.no_grad
    def transform(self, target, vars):
        return [t.copysign_(g) for t,g in zip(target, vars.get_grad())]

class UpdateSign(Transform):
    """use per-weight magnitudes from grad while using sign from update."""
    def __init__(self, target: Target = 'update'):
        super().__init__(target=target)

    @torch.no_grad
    def transform(self, target, vars):
        return [g.copysign(t) for t,g in zip(target, vars.get_grad())] # no in-place

class GraftToGrad(Transform):
    """use gradient norm and update direction."""
    def __init__(self, tensorwise:bool=False, ord:float=2, eps:float = 1e-6, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(self.defaults)
        return TensorList(target).graft_(vars.get_grad(), tensorwise=tensorwise, ord=ord, eps=eps)

class GraftGradToUpdate(Transform):
    """use update norm and gradient direction."""
    def __init__(self, tensorwise:bool=False, ord:float=2, eps:float = 1e-6, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(self.defaults)
        return TensorList(vars.get_grad()).graft(target, tensorwise=tensorwise, ord=ord, eps=eps)


class GraftToParams(Transform):
    """makes update norm be set to parameter norm, but norm won't go below eps"""
    def __init__(self, tensorwise:bool=False, ord:float=2, eps:float = 1e-4, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(self.defaults)
        return TensorList(target).graft_(vars.params, tensorwise=tensorwise, ord=ord, eps=eps)

class Relative(Transform):
    """multiplies update by absolute parameter values to make it relative to their magnitude, min_value is minimum value to avoid getting stuck at 0"""
    def __init__(self, min_value:float = 1e-4, target: Target = 'update'):
        defaults = dict(min_value=min_value)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        mul = TensorList(vars.params).abs().clamp_(self.get_settings('min_value', params=vars))
        torch._foreach_mul_(target, mul)
        return target

class FillLoss(Transform):
    """returns update filled with loss value times alpha"""
    def __init__(self, alpha: float = 1, backward: bool = True, target: "Target" = 'update'):
        defaults = dict(alpha=alpha, backward=backward)
        super().__init__(defaults, target=target)

    def transform(self, target, vars):
        alpha = self.get_settings('alpha', params=vars)
        loss = vars.get_loss(backward=self.defaults['backward'])
        return [t.fill_(loss*a) for t,a in zip(target, alpha)]

class MulByLoss(Transform):
    """multiplies update by loss times alpha"""
    def __init__(self, alpha: float = 1, min_value:float = 1e-8, backward: bool = True, target: Target = 'update'):
        defaults = dict(alpha=alpha, min_value=min_value, backward=backward)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        alpha, min_value = self.get_settings('alpha', 'min_value', params=vars)
        loss = vars.get_loss(backward=self.defaults['backward'])
        mul = [max(loss*a, mv) for a,mv in zip(alpha, min_value)]
        torch._foreach_mul_(target, mul)
        return target

class DivByLoss(Transform):
    """divides update by loss times alpha"""
    def __init__(self, alpha: float = 1, min_value:float = 1e-8, backward: bool = True, target: Target = 'update'):
        defaults = dict(alpha=alpha, min_value=min_value, backward=backward)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        alpha, min_value = self.get_settings('alpha', 'min_value', params=vars)
        loss = vars.get_loss(backward=self.defaults['backward'])
        mul = [max(loss*a, mv) for a,mv in zip(alpha, min_value)]
        torch._foreach_div_(target, mul)
        return target



def _sequential_step(self: Module, vars: Vars, sequential: bool):
    params = vars.params
    steps = self.defaults['steps']

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
            if (not new_vars.stop) and (new_vars.update is not None):
                if new_vars.last_module_lrs is not None:
                    torch._foreach_mul_(new_vars.update, new_vars.last_module_lrs)

                torch._foreach_sub_(params, new_vars.update)

            # create new vars since we are at a new point, that means grad, update and loss will be None
            new_vars = Vars(params=new_vars.params, closure=new_vars.closure,
                            model=new_vars.model, current_step=new_vars.current_step + 1)

            # step
            new_vars = m.step(new_vars)

        # final parameter update
        if (not new_vars.stop) and (new_vars.update is not None):
            if new_vars.last_module_lrs is not None:
                torch._foreach_mul_(new_vars.update, new_vars.last_module_lrs)

            torch._foreach_sub_(params, new_vars.update)

    # if last module, update is applied so return new vars
    if params_before_steps is None:
        new_vars.stop = True
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