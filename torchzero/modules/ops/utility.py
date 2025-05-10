from collections import deque

import torch

from ...core import Module, Target, Transform
from ...utils.tensorlist import Distributions, TensorList


class Clone(Transform):
    def __init__(self): super().__init__({}, uses_grad=False)
    @torch.no_grad
    def transform(self, tensors, params, grads, vars): return [t.clone() for t in tensors]

class Grad(Module):
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def step(self, vars):
        vars.update = [g.clone() for g in vars.get_grad()]
        return vars

class Params(Module):
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def step(self, vars):
        vars.update = [p.clone() for p in vars.params]
        return vars

class Update(Module):
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def step(self, vars):
        vars.update = [u.clone() for u in vars.get_update()]
        return vars

class Zeros(Module):
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def step(self, vars):
        vars.update = [torch.zeros_like(p) for p in vars.params]
        return vars

class Ones(Module):
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def step(self, vars):
        vars.update = [torch.ones_like(p) for p in vars.params]
        return vars

class Fill(Module):
    def __init__(self, value: float):
        defaults = dict(value=value)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, vars):
        vars.update = [torch.full_like(p, self.settings[p]['value']) for p in vars.params]
        return vars

class RandomSample(Module):
    def __init__(self, eps: float = 1, distribution: Distributions = 'normal'):
        defaults = dict(eps=eps, distribution=distribution)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, vars):
        vars.update = TensorList(vars.params).sample_like(
            eps=self.get_settings('eps',params=vars.params), distribution=self.settings[vars.params[0]]['distribution']
        )
        return vars

class Randn(Module):
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def step(self, vars):
        vars.update = [torch.randn_like(p) for p in vars.params]
        return vars

class Uniform(Module):
    def __init__(self, low: float, high: float):
        defaults = dict(low=low, high=high)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, vars):
        low,high = self.get_settings('low','high', params=vars.params)
        vars.update = [torch.empty_like(t).uniform_(l,h) for t,l,h in zip(vars.params, low, high)]
        return vars

class GradToNone(Module):
    def __init__(self): super().__init__()
    def step(self, vars):
        vars.grad = None
        return vars

class UpdateToNone(Module):
    def __init__(self): super().__init__()
    def step(self, vars):
        vars.update = None
        return vars

class Identity(Module):
    def __init__(self, *args, **kwargs): super().__init__()
    def step(self, vars): return vars

NoOp = Identity