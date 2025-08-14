from collections import deque
from collections.abc import Iterable, Sequence
from functools import partial
from operator import itemgetter
from typing import Literal

import torch

from ...core import Chainable, Module, Target, TensorwiseTransform, Transform, Var
from ...utils import (
    Distributions,
    Metrics,
    NumberList,
    TensorList,
    set_storage_,
    tofloat,
    unpack_dicts,
    unpack_states,
)


class Previous(TensorwiseTransform):
    """Maintains an update from n steps back, for example if n=1, returns previous update"""
    def __init__(self, n=1, target: Target = 'update'):
        defaults = dict(n=n)
        super().__init__(uses_grad=False, defaults=defaults, target=target)


    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, loss, state, setting):
        n = setting['n']

        if 'history' not in state:
            state['history'] = deque(maxlen=n+1)

        state['history'].append(tensor)

        return state['history'][0]


class LastDifference(Transform):
    """Outputs difference between past two updates."""
    def __init__(self,target: Target = 'update'):
        super().__init__({}, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        prev_tensors = unpack_states(states, tensors, 'prev_tensors') # initialized to 0
        difference = torch._foreach_sub(tensors, prev_tensors)
        for p, c in zip(prev_tensors, tensors): p.set_(c)
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
        for p, c in zip(prev_grad, grad): p.copy_(c)
        var.update = list(difference)
        return var

class LastParamDifference(Module):
    """Outputs difference between past two parameters, which is the effective previous update."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def step(self, var):
        params = var.params
        prev_params = self.get_state(var.params, 'prev_params') # initialized to 0
        difference = torch._foreach_sub(params, prev_params)
        for p, c in zip(prev_params, params): p.copy_(c)
        var.update = list(difference)
        return var



class LastProduct(Transform):
    """Outputs difference between past two updates."""
    def __init__(self,target: Target = 'update'):
        super().__init__({}, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
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
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
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
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
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
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        return [t.copysign_(g) for t,g in zip(tensors, grads)]

class UpdateSign(Transform):
    """Outputs gradient with sign copied from the update."""
    def __init__(self, target: Target = 'update'):
        super().__init__({}, uses_grad=True, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        return [g.copysign(t) for t,g in zip(tensors, grads)] # no in-place

class GraftToGrad(Transform):
    """Grafts update to the gradient, that is update is rescaled to have the same norm as the gradient."""
    def __init__(self, tensorwise:bool=False, ord:Metrics=2, eps:float = 1e-6, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=True, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(settings[0])
        return TensorList(tensors).graft_(grads, tensorwise=tensorwise, ord=ord, eps=eps)

class GraftGradToUpdate(Transform):
    """Outputs gradient grafted to update, that is gradient rescaled to have the same norm as the update."""
    def __init__(self, tensorwise:bool=False, ord:Metrics=2, eps:float = 1e-6, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=True, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(settings[0])
        return TensorList(grads).graft(tensors, tensorwise=tensorwise, ord=ord, eps=eps)


class GraftToParams(Transform):
    """Grafts update to the parameters, that is update is rescaled to have the same norm as the parameters, but no smaller than :code:`eps`."""
    def __init__(self, tensorwise:bool=False, ord:Metrics=2, eps:float = 1e-4, target: Target = 'update'):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(settings[0])
        return TensorList(tensors).graft_(params, tensorwise=tensorwise, ord=ord, eps=eps)

class Relative(Transform):
    """Multiplies update by absolute parameter values to make it relative to their magnitude, :code:`min_value` is minimum allowed value to avoid getting stuck at 0."""
    def __init__(self, min_value:float = 1e-4, target: Target = 'update'):
        defaults = dict(min_value=min_value)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
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
        loss = var.get_loss(backward=self.defaults['backward'])
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
        loss = var.get_loss(backward=self.defaults['backward'])
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
        loss = var.get_loss(backward=self.defaults['backward'])
        mul = [max(loss*a, mv) for a,mv in zip(alpha, min_value)]
        torch._foreach_div_(var.update, mul)
        return var


class NoiseSign(Transform):
    """Outputs random tensors with sign copied from the update."""
    def __init__(self, distribution:Distributions = 'normal', variance:float | None = None):
        defaults = dict(distribution=distribution, variance=variance)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        variance = unpack_dicts(settings, 'variance')
        return TensorList(tensors).sample_like(settings[0]['distribution'], variance=variance).copysign_(tensors)

class HpuEstimate(Transform):
    """returns ``y/||s||``, where ``y`` is difference between current and previous update (gradient), ``s`` is difference between current and previous parameters. The returned tensors are a finite difference approximation to hessian times previous update."""
    def __init__(self):
        defaults = dict()
        super().__init__(defaults, uses_grad=False)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_params', 'prev_update')

    @torch.no_grad
    def update_tensors(self, tensors, params, grads, loss, states, settings):
        prev_params, prev_update = self.get_state(params, 'prev_params', 'prev_update') # initialized to 0
        s = torch._foreach_sub(params, prev_params)
        y = torch._foreach_sub(tensors, prev_update)
        for p, c in zip(prev_params, params): p.copy_(c)
        for p, c in zip(prev_update, tensors): p.copy_(c)
        torch._foreach_div_(y, torch.linalg.norm(torch.cat([t.ravel() for t in s])).clip(min=1e-8)) # pylint:disable=not-callable
        self.store(params, 'y', y)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        return [self.state[p]['y'] for p in params]

class RandomHvp(Module):
    """Returns a hessian-vector product with a random vector"""

    def __init__(
        self,
        n_samples: int = 1,
        distribution: Distributions = "normal",
        update_freq: int = 1,
        hvp_method: Literal["autograd", "forward", "central"] = "autograd",
        h=1e-3,
    ):
        defaults = dict(n_samples=n_samples, distribution=distribution, hvp_method=hvp_method, h=h, update_freq=update_freq)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)
        settings = self.settings[params[0]]
        n_samples = settings['n_samples']
        distribution = settings['distribution']
        hvp_method = settings['hvp_method']
        h = settings['h']
        update_freq = settings['update_freq']

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        D = None
        if step % update_freq == 0:

            rgrad = None
            for i in range(n_samples):
                u = params.sample_like(distribution=distribution, variance=1)

                Hvp, rgrad = self.Hvp(u, at_x0=True, var=var, rgrad=rgrad, hvp_method=hvp_method,
                                    h=h, normalize=True, retain_grad=i < n_samples-1)

                if D is None: D = Hvp
                else: torch._foreach_add_(D, Hvp)

            if n_samples > 1: torch._foreach_div_(D, n_samples)
            if update_freq != 1:
                assert D is not None
                D_buf = self.get_state(params, "D", cls=TensorList)
                D_buf.set_(D)

        if D is None:
            D = self.get_state(params, "D", cls=TensorList)

        var.update = list(D)
        return var

@torch.no_grad
def _load_best_parameters(params: Sequence[torch.Tensor], best_params: Sequence[torch.Tensor]):
    for p_cur, p_best in zip(params, best_params):
        set_storage_(p_cur, p_best)

class SaveBest(Module):
    """Saves best parameters found so far, ones that have lowest loss. Put this as the last module.

    Adds the following attrs:

    - ``best_params`` - a list of tensors with best parameters.
    - ``best_loss`` - loss value with ``best_params``.
    - ``load_best_parameters`` - a function that sets parameters to the best parameters./

    ## Examples
    ```python
    def rosenbrock(x, y):
        return (1 - x)**2 + (100 * (y - x**2))**2

    xy = torch.tensor((-1.1, 2.5), requires_grad=True)
    opt = tz.Modular(
        [xy],
        tz.m.NAG(0.999),
        tz.m.LR(1e-6),
        tz.m.SaveBest()
    )

    # optimize for 1000 steps
    for i in range(1000):
        loss = rosenbrock(*xy)
        opt.zero_grad()
        loss.backward()
        opt.step(loss=loss) # SaveBest needs closure or loss

    # NAG overshot, but we saved the best params
    print(f'{rosenbrock(*xy) = }') # >> 3.6583
    print(f"{opt.attrs['best_loss'] = }") # >> 0.000627

    # load best parameters
    opt.attrs['load_best_params']()
    print(f'{rosenbrock(*xy) = }') # >> 0.000627
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad
    def step(self, var):
        loss = tofloat(var.get_loss(False))
        lowest_loss = self.global_state.get('lowest_loss', float("inf"))

        if loss < lowest_loss:
            self.global_state['lowest_loss'] = loss
            best_params = var.attrs['best_params'] = [p.clone() for p in var.params]
            var.attrs['best_loss'] = loss
            var.attrs['load_best_params'] = partial(_load_best_parameters, params=var.params, best_params=best_params)

        return var
