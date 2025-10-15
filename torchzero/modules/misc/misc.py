from collections import deque
from collections.abc import Iterable, Sequence
from functools import partial
from operator import itemgetter
from typing import Literal

import torch

from ...core import Chainable, Module,  TensorTransform, Transform, Objective
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


class Previous(TensorTransform):
    """Maintains an update from n steps back, for example if n=1, returns previous update"""
    def __init__(self, n=1):
        defaults = dict(n=n)
        super().__init__(defaults=defaults)

        self.add_projected_keys("grad", "history")

    @torch.no_grad
    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        n = setting['n']

        if 'history' not in state:
            state['history'] = deque(maxlen=n+1)

        state['history'].append(tensor)

        return state['history'][0]


class LastDifference(TensorTransform):
    """Outputs difference between past two updates."""
    def __init__(self,):
        super().__init__()
        self.add_projected_keys("grad", "prev_tensors")

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        prev_tensors = unpack_states(states, tensors, 'prev_tensors') # initialized to 0
        difference = torch._foreach_sub(tensors, prev_tensors)
        for p, c in zip(prev_tensors, tensors): p.set_(c)
        return difference

class LastGradDifference(Module):
    """Outputs difference between past two gradients."""
    def __init__(self):
        super().__init__()
        self.add_projected_keys("grad", "prev_grad")

    @torch.no_grad
    def apply(self, objective):
        grad = objective.get_grads()
        prev_grad = self.get_state(objective.params, 'prev_grad') # initialized to 0
        difference = torch._foreach_sub(grad, prev_grad)
        for p, c in zip(prev_grad, grad): p.copy_(c)
        objective.updates = list(difference)
        return objective

class LastParamDifference(Module):
    """Outputs difference between past two parameters, which is the effective previous update."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def apply(self, objective):
        params = objective.params
        prev_params = self.get_state(objective.params, 'prev_params') # initialized to 0
        difference = torch._foreach_sub(params, prev_params)
        for p, c in zip(prev_params, params): p.copy_(c)
        objective.updates = list(difference)
        return objective



class LastProduct(TensorTransform):
    """Outputs difference between past two updates."""
    def __init__(self):
        super().__init__()
        self.add_projected_keys("grad", "prev")

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        prev = unpack_states(states, tensors, 'prev', init=torch.ones_like) # initialized to 1 for prod
        prod = torch._foreach_mul(tensors, prev)
        for p, c in zip(prev, tensors): p.set_(c)
        return prod

class LastRatio(TensorTransform):
    """Outputs ratio between past two updates, the numerator is determined by ``numerator`` argument."""
    def __init__(self, numerator: Literal['cur', 'prev'] = 'cur'):
        defaults = dict(numerator=numerator)
        super().__init__(defaults)
        self.add_projected_keys("grad", "prev")

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        prev = unpack_states(states, tensors, 'prev', init = torch.ones_like) # initialized to ones
        numerator = settings[0]['numerator']
        if numerator == 'cur': ratio = torch._foreach_div(tensors, prev)
        else: ratio = torch._foreach_div(prev, tensors)
        for p, c in zip(prev, tensors): p.set_(c)
        return ratio

class LastAbsoluteRatio(TensorTransform):
    """Outputs ratio between absolute values of past two updates the numerator is determined by ``numerator`` argument."""
    def __init__(self, numerator: Literal['cur', 'prev'] = 'cur', eps:float=1e-8):
        defaults = dict(numerator=numerator, eps=eps)
        super().__init__(defaults)
        self.add_projected_keys("grad", "prev")

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        prev = unpack_states(states, tensors, 'prev', init = torch.ones_like) # initialized to ones
        numerator = settings[0]['numerator']
        eps = NumberList(s['eps'] for s in settings)

        torch._foreach_abs_(tensors)
        torch._foreach_clamp_min_(prev, eps)

        if numerator == 'cur': ratio = torch._foreach_div(tensors, prev)
        else: ratio = torch._foreach_div(prev, tensors)
        for p, c in zip(prev, tensors): p.set_(c)
        return ratio

class GradSign(TensorTransform):
    """Copies gradient sign to update."""
    def __init__(self):
        super().__init__(uses_grad=True)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        return [t.copysign_(g) for t,g in zip(tensors, grads)]

class UpdateSign(TensorTransform):
    """Outputs gradient with sign copied from the update."""
    def __init__(self):
        super().__init__(uses_grad=True)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        return [g.copysign(t) for t,g in zip(tensors, grads)] # no in-place

class GraftToGrad(TensorTransform):
    """Grafts update to the gradient, that is update is rescaled to have the same norm as the gradient."""
    def __init__(self, tensorwise:bool=False, ord:Metrics=2, eps:float = 1e-6):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=True)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(settings[0])
        return TensorList(tensors).graft_(grads, tensorwise=tensorwise, ord=ord, eps=eps)

class GraftGradToUpdate(TensorTransform):
    """Outputs gradient grafted to update, that is gradient rescaled to have the same norm as the update."""
    def __init__(self, tensorwise:bool=False, ord:Metrics=2, eps:float = 1e-6):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, uses_grad=True)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(settings[0])
        return TensorList(grads).graft(tensors, tensorwise=tensorwise, ord=ord, eps=eps)


class GraftToParams(TensorTransform):
    """Grafts update to the parameters, that is update is rescaled to have the same norm as the parameters, but no smaller than ``eps``."""
    def __init__(self, tensorwise:bool=False, ord:Metrics=2, eps:float = 1e-4):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(settings[0])
        return TensorList(tensors).graft_(params, tensorwise=tensorwise, ord=ord, eps=eps)

class Relative(TensorTransform):
    """Multiplies update by absolute parameter values to make it relative to their magnitude, ``min_value`` is minimum allowed value to avoid getting stuck at 0."""
    def __init__(self, min_value:float = 1e-4):
        defaults = dict(min_value=min_value)
        super().__init__(defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        mul = TensorList(params).abs().clamp_([s['min_value'] for s in settings])
        torch._foreach_mul_(tensors, mul)
        return tensors

class FillLoss(Module):
    """Outputs tensors filled with loss value times ``alpha``"""
    def __init__(self, alpha: float = 1, backward: bool = True):
        defaults = dict(alpha=alpha, backward=backward)
        super().__init__(defaults)

    @torch.no_grad
    def apply(self, objective):
        alpha = self.get_settings(objective.params, 'alpha')
        loss = objective.get_loss(backward=self.defaults['backward'])
        objective.updates = [torch.full_like(p, loss*a) for p,a in zip(objective.params, alpha)]
        return objective

class MulByLoss(TensorTransform):
    """Multiplies update by loss times ``alpha``"""
    def __init__(self, alpha: float = 1, min_value:float = 1e-16, backward: bool = True):
        defaults = dict(alpha=alpha, min_value=min_value, backward=backward)
        super().__init__(defaults, uses_loss=True)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        assert loss is not None
        alpha, min_value = unpack_dicts(settings, 'alpha', 'min_value')
        mul = [max(loss*a, mv) for a,mv in zip(alpha, min_value)]
        torch._foreach_mul_(tensors, mul)
        return tensors

class DivByLoss(TensorTransform):
    """Divides update by loss times ``alpha``"""
    def __init__(self, alpha: float = 1, min_value:float = 1e-16, backward: bool = True):
        defaults = dict(alpha=alpha, min_value=min_value, backward=backward)
        super().__init__(defaults, uses_loss=True)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        assert loss is not None
        alpha, min_value = unpack_dicts(settings, 'alpha', 'min_value')
        denom = [max(loss*a, mv) for a,mv in zip(alpha, min_value)]
        torch._foreach_div_(tensors, denom)
        return tensors


class NoiseSign(TensorTransform):
    """Outputs random tensors with sign copied from the update."""
    def __init__(self, distribution:Distributions = 'normal', variance:float | None = None):
        defaults = dict(distribution=distribution, variance=variance)
        super().__init__(defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        variance = unpack_dicts(settings, 'variance')
        return TensorList(tensors).sample_like(settings[0]['distribution'], variance=variance).copysign_(tensors)

class HpuEstimate(TensorTransform):
    """returns ``y/||s||``, where ``y`` is difference between current and previous update (gradient), ``s`` is difference between current and previous parameters. The returned tensors are a finite difference approximation to hessian times previous update."""
    def __init__(self):
        defaults = dict()
        super().__init__(defaults)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_params', 'prev_update')

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        prev_params, prev_update = self.get_state(params, 'prev_params', 'prev_update') # initialized to 0
        s = torch._foreach_sub(params, prev_params)
        y = torch._foreach_sub(tensors, prev_update)
        for p, c in zip(prev_params, params): p.copy_(c)
        for p, c in zip(prev_update, tensors): p.copy_(c)
        torch._foreach_div_(y, torch.linalg.norm(torch.cat([t.ravel() for t in s])).clip(min=1e-8)) # pylint:disable=not-callable
        self.store(params, 'y', y)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        return [self.state[p]['y'] for p in params]

class RandomHvp(Module):
    """Returns a hessian-vector product with a random vector, optionally times vector"""

    def __init__(
        self,
        n_samples: int = 1,
        distribution: Distributions = "normal",
        update_freq: int = 1,
        zHz: bool = False,
        hvp_method: Literal["autograd", "fd_forward", "central"] = "autograd",
        h=1e-3,
        seed: int | None = None
    ):
        defaults = locals().copy()
        del defaults['self']
        super().__init__(defaults)

    @torch.no_grad
    def apply(self, objective):
        params = TensorList(objective.params)

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        D = None
        update_freq = self.defaults['update_freq']
        if step % update_freq == 0:

            D, _ = objective.hutchinson_hessian(
                rgrad = None,
                at_x0 = True,
                n_samples = self.defaults['n_samples'],
                distribution = self.defaults['distribution'],
                hvp_method = self.defaults['hvp_method'],
                h = self.defaults['h'],
                zHz = self.defaults["zHz"],
                generator = self.get_generator(params[0].device, self.defaults["seed"]),
            )

            if update_freq != 1:
                assert D is not None
                D_buf = self.get_state(params, "D", cls=TensorList)
                D_buf.set_(D)

        if D is None:
            D = self.get_state(params, "D", cls=TensorList)

        objective.updates = list(D)
        return objective

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
    opt = tz.Optimizer(
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
    def apply(self, objective):
        loss = tofloat(objective.get_loss(False))
        lowest_loss = self.global_state.get('lowest_loss', float("inf"))

        if loss < lowest_loss:
            self.global_state['lowest_loss'] = loss
            best_params = objective.attrs['best_params'] = [p.clone() for p in objective.params]
            objective.attrs['best_loss'] = loss
            objective.attrs['load_best_params'] = partial(_load_best_parameters, params=objective.params, best_params=best_params)

        return objective
