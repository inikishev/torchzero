from collections import deque

import torch

from ...core import ParameterwiseTransform, Target, Transform, Module
from ...utils.tensorlist import Distributions, TensorList


class Clone(Transform):
    def __init__(self): super().__init__()
    @torch.no_grad
    def transform(self, target, vars): return [t.clone() for t in target]

class Grad(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__(target=target)
    @torch.no_grad
    def transform(self, target, vars): return [g.clone() for g in vars.get_grad()]

class Params(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__(target=target)
    @torch.no_grad
    def transform(self, target, vars): return [p.clone() for p in vars.params]

class Update(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__(target=target)
    @torch.no_grad
    def transform(self, target, vars): return [u.clone() for u in vars.get_update()]

class Zeros(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__(target=target)
    @torch.no_grad
    def transform(self, target, vars):
        torch._foreach_zero_(target)
        return target

class Ones(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__(target=target)
    @torch.no_grad
    def transform(self, target, vars): return [t.fill_(1) for t in target]

class Fill(Transform):
    def __init__(self, value: float, target: "Target" = 'update'):
        defaults = dict(value=value)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        return [t.fill_(v) for t,v in zip(target, self.get_settings('value', params=vars))]

class RandomSample(Transform):
    def __init__(self, eps: float = 1, distribution: Distributions = 'normal', target: "Target" = 'update'):
        defaults = dict(eps=eps, distribution=distribution)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        return TensorList(target).sample_like(eps=self.get_settings('value',params=vars), distribution=self.defaults['distribution'])

class Randn(Transform):
    def __init__(self, target: "Target" = 'update'):
        super().__init__({}, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        return [torch.randn_like(t) for t in target]

class Uniform(Transform):
    def __init__(self, low: float, high: float, target: "Target" = 'update'):
        defaults = dict(low=low, high=high)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        low,high = self.get_settings('low','high', params=vars)
        return [t.uniform_(l,h) for t,l,h in zip(target, low, high)]


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