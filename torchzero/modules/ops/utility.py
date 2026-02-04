from collections import deque

import torch

from ...core import Module,  Transform
from ...utils.tensorlist import Distributions, TensorList
from ...linalg.linear_operator import ScaledIdentity

class Clone(Module):
    """Clones input. May be useful to store some intermediate result and make sure it doesn't get affected by in-place operations"""
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def apply(self, objective):
        objective.updates = [u.clone() for u in objective.get_updates()]
        return objective

class Grad(Module):
    """Outputs the gradient"""
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def apply(self, objective):
        objective.updates = [g.clone() for g in objective.get_grads()]
        return objective

class Params(Module):
    """Outputs parameters"""
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def apply(self, objective):
        objective.updates = [p.clone() for p in objective.params]
        return objective

class Zeros(Module):
    """Outputs zeros"""
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def apply(self, objective):
        objective.updates = [torch.zeros_like(p) for p in objective.params]
        return objective

class Ones(Module):
    """Outputs ones"""
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def apply(self, objective):
        objective.updates = [torch.ones_like(p) for p in objective.params]
        return objective

class Fill(Module):
    """Outputs tensors filled with ``value``"""
    def __init__(self, value: float):
        defaults = dict(value=value)
        super().__init__(defaults)

    @torch.no_grad
    def apply(self, objective):
        objective.updates = [torch.full_like(p, self.settings[p]['value']) for p in objective.params]
        return objective

class RandomSample(Module):
    """Outputs tensors filled with random numbers from distribution depending on value of ``distribution``."""
    def __init__(self, distribution: Distributions = 'normal', variance:float | None = None):
        defaults = dict(distribution=distribution, variance=variance)
        super().__init__(defaults)

    @torch.no_grad
    def apply(self, objective):
        distribution = self.defaults['distribution']
        variance = self.get_settings(objective.params, 'variance')
        objective.updates = TensorList(objective.params).sample_like(distribution=distribution, variance=variance)
        return objective

class Randn(Module):
    """Outputs tensors filled with random numbers from a normal distribution with mean 0 and variance 1."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def apply(self, objective):
        objective.updates = [torch.randn_like(p) for p in objective.params]
        return objective

class Uniform(Module):
    """Outputs tensors filled with random numbers from uniform distribution between ``low`` and ``high``."""
    def __init__(self, low: float, high: float):
        defaults = dict(low=low, high=high)
        super().__init__(defaults)

    @torch.no_grad
    def apply(self, objective):
        low,high = self.get_settings(objective.params, 'low','high')
        objective.updates = [torch.empty_like(t).uniform_(l,h) for t,l,h in zip(objective.params, low, high)]
        return objective

class GradToNone(Module):
    """Sets ``grad`` attribute to None on ``objective``."""
    def __init__(self): super().__init__()
    def apply(self, objective):
        objective.grads = None
        return objective

class UpdateToNone(Module):
    """Sets ``update`` attribute to None on ``var``."""
    def __init__(self): super().__init__()
    def apply(self, objective):
        objective.updates = None
        return objective

class Identity(Module):
    """Identity operator that is argument-insensitive. This also can be used as identity hessian for trust region methods."""
    def __init__(self, *args, **kwargs): super().__init__()
    def update(self, objective): pass
    def apply(self, objective): return objective
    def get_H(self, objective):
        n = sum(p.numel() for p in objective.params)
        p = objective.params[0]
        return ScaledIdentity(shape=(n,n), device=p.device, dtype=p.dtype)

Noop = Identity
"""A placeholder identity operator that is argument-insensitive."""
