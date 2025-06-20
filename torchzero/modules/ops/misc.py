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
    """Outputs difference between past two updates."""
    def __init__(self,target: Target = 'update'):
        super().__init__({}, uses_grad=False, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        prev = unpack_states(states, tensors, 'prev_target') # initialized to 0
        difference = torch._foreach_sub(tensors, prev)
        for p, c in zip(prev, tensors): p.set_(c)
        return difference

class LastGradDifference(Module):
    """Outputs difference between past two gradients."""
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
    """Outputs difference between past two updates."""
    def __init__(self,target: Target = 'update'):
        super().__init__({}, uses_grad=False, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        prev = unpack_states(states, tensors, 'prev', init=torch.ones_like) # initialized to 1 for prod
        prod = torch._foreach_mul(tensors, prev)
        for p, c in zip(prev, tensors): p.set_(c)
        return prod

class LastRatio(Transform):
    """Outputs ratio between past two updates, the numerator is determined by :code:`numerator` argument."""
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
    """Outputs ratio between absolute values of past two updates the numerator is determined by :code:`numerator` argument."""
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
    """Copies gradient sign to update."""
    def __init__(self, target: Target = 'update'):
        super().__init__({}, uses_grad=True, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        return [t.copysign_(g) for t,g in zip(tensors, grads)]

class UpdateSign(Transform):
    """Outputs gradient with sign copied from the update."""
    def __init__(self, target: Target = 'update'):
        super().__init__({}, uses_grad=True, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        return [g.copysign(t) for t,g in zip(tensors, grads)] # no in-place

class GraftToGrad(Transform):
    """Grafts update to the gradient, that is update is rescaled to have the same norm as the gradient."""
    def __init__(self, tensorwise:bool=False, ord:float=2, eps:float = 1e-6, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=True, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(settings[0])
        return TensorList(tensors).graft_(grads, tensorwise=tensorwise, ord=ord, eps=eps)

class GraftGradToUpdate(Transform):
    """Outputs gradient grafted to update, that is gradient rescaled to have the same norm as the update."""
    def __init__(self, tensorwise:bool=False, ord:float=2, eps:float = 1e-6, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=True, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(settings[0])
        return TensorList(grads).graft(tensors, tensorwise=tensorwise, ord=ord, eps=eps)


class GraftToParams(Transform):
    """Grafts update to the parameters, that is update is rescaled to have the same norm as the parameters, but no smaller than :code:`eps`."""
    def __init__(self, tensorwise:bool=False, ord:float=2, eps:float = 1e-4, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(settings[0])
        return TensorList(tensors).graft_(params, tensorwise=tensorwise, ord=ord, eps=eps)

class Relative(Transform):
    """Multiplies update by absolute parameter values to make it relative to their magnitude, :code:`min_value` is minimum allowed value to avoid getting stuck at 0."""
    def __init__(self, min_value:float = 1e-4, target: Target = 'update'):
        defaults = dict(min_value=min_value)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        mul = TensorList(params).abs().clamp_([s['min_value'] for s in settings])
        torch._foreach_mul_(tensors, mul)
        return tensors

class FillLoss(Module):
    """Outputs tensors filled with loss value times :code:`alpha`"""
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
    """Multiplies update by loss times :code:`alpha`"""
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
    """Divides update by loss times :code:`alpha`"""
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

    if sequential: modules = self.get_children_sequence() * steps
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
    """Performs :code:`steps` inner steps with :code:`module` per each step.

    The update is taken to be the parameter difference between parameters before and after the inner loop."""
    def __init__(self, module: Chainable, steps: int):
        defaults = dict(steps=steps)
        super().__init__(defaults)
        self.set_child('module', module)

    @torch.no_grad
    def step(self, var):
        return _sequential_step(self, var, sequential=False)

class Sequential(Module):
    """On each step, this sequentially steps with :code:`modules` :code:`steps` times.

    The update is taken to be the parameter difference between parameters before and after the inner loop."""
    def __init__(self, modules: Iterable[Chainable], steps: int=1):
        defaults = dict(steps=steps)
        super().__init__(defaults)
        self.set_children_sequence(modules)

    @torch.no_grad
    def step(self, var):
        return _sequential_step(self, var, sequential=True)


class GradientAccumulation(Module):
    """Uses :code:`n` steps to accumulate gradients, after :code:`n` gradients have been accumulated, they are passed to :code:`modules` and parameters are updates.

    Accumulating gradients for :code:`n` steps is equivalent to increasing batch size by :code:`n`. Increasing the batch size
    is more computationally efficient, but sometimes it is not feasible due to memory constraints.

    .. note::
        Technically this can accumulate any inputs, including updates generated by previous modules. As long as this module is first, it will accumulate the gradients.

    Args:
        modules (Chainable): modules that perform a step every :code:`n` steps using the accumulated gradients.
        n (int): number of gradients to accumulate.
        mean (bool, optional): if True, uses mean of accumulated gradients, otherwise uses sum. Defaults to True.
        stop (bool, optional):
            this module prevents next modules from stepping unless :code:`n` gradients have been accumulate. Setting this argument to False disables that. Defaults to True.

    Examples:
        Adam with gradients accumulated for 16 batches.

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.GradientAccumulation(
                    modules=[tz.m.Adam(), tz.m.LR(1e-2)],
                    n=16
                )
            )

    """
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
    """Applies dropout to the update.

    For each weight the update to that weight has :code:`p` probability to be set to 0.
    This can be used to implement gradient dropout or update dropout depending on placement.

    Args:
        p (float, optional): probability that update for a weight is replaced with 0. Defaults to 0.5.
        graft (bool, optional):
            if True, update after dropout is rescaled to have the same norm as before dropout. Defaults to False.
        target (Target, optional): what to set on var, refer to documentation. Defaults to 'update'.


    Examples:
        Gradient dropout.

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.Dropout(0.5),
                tz.m.Adam(),
                tz.m.LR(1e-3)
            )

        Update dropout.

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.Adam(),
                tz.m.Dropout(0.5),
                tz.m.LR(1e-3)
            )

    """
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

def _bernoulli_like(tensor, p = 0.5, generator = None):
    """p is probability of a 1, other values will be 0."""
    return torch.bernoulli(torch.full_like(tensor, p), generator = generator)

class WeightDropout(Module):
    """
    Changes the closure so that it evaluates loss and gradients with random weights replaced with 0.

    Dropout can be disabled for a parameter by setting :code:`use_dropout=False` in corresponding parameter group.

    Args:
        p (float, optional): probability that any weight is replaced with 0. Defaults to 0.5.
        graft (bool, optional):
            if True, parameters after dropout are rescaled to have the same norm as before dropout. Defaults to False.
    """
    def __init__(self, p: float = 0.5, graft: bool = True):
        defaults = dict(p=p, graft=graft, use_dropout=True)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        closure = var.closure
        if closure is None: raise RuntimeError('WeightDropout requires closure')
        params = TensorList(var.params)
        p = NumberList(self.settings[p]['p'] for p in params)

        # create masks
        mask = []
        for p, m in zip(params, mask):
            prob = self.settings[p]['p']
            use_dropout = self.settings[p]['use_dropout']
            if use_dropout: mask.append(_bernoulli_like(p, prob))
            else: mask.append(torch.ones_like(p))

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
    """Outputs random tensors with sign copied from the update."""
    def __init__(self, distribution:Distributions = 'normal', alpha = 1):
        defaults = dict(distribution=distribution, alpha=alpha)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        alpha = [s['alpha'] for s in settings]
        distribution = self.settings[params[0]]['distribution']
        return TensorList(tensors).sample_like(alpha, distribution).copysign_(tensors)


class NegateOnLossIncrease(Module):
    """Uses an extra forward pass to evaluate loss at :code:`parameters+update`,
    if loss is larger than at :code:`parameters`,
    the update is set to 0 if :code:`backtrack=False` and to :code:`-update` otherwise"""
    def __init__(self, backtrack=False):
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