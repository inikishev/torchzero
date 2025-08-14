from collections import deque

import torch

from ...core import Module, Target, Transform
from ...utils.tensorlist import Distributions, TensorList
from ...utils.linalg.linear_operator import ScaledIdentity

class Clone(Module):
    """Clones input. May be useful to store some intermediate result and make sure it doesn't get affected by in-place operations"""
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def step(self, var):
        var.update = [u.clone() for u in var.get_update()]
        return var

class Grad(Module):
    """Outputs the gradient"""
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def step(self, var):
        var.update = [g.clone() for g in var.get_grad()]
        return var

class Params(Module):
    """Outputs parameters"""
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def step(self, var):
        var.update = [p.clone() for p in var.params]
        return var

class Zeros(Module):
    """Outputs zeros"""
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def step(self, var):
        var.update = [torch.zeros_like(p) for p in var.params]
        return var

class Ones(Module):
    """Outputs ones"""
    def __init__(self):
        super().__init__({})
    @torch.no_grad
    def step(self, var):
        var.update = [torch.ones_like(p) for p in var.params]
        return var

class Fill(Module):
    """Outputs tensors filled with :code:`value`"""
    def __init__(self, value: float):
        defaults = dict(value=value)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        var.update = [torch.full_like(p, self.settings[p]['value']) for p in var.params]
        return var

class RandomSample(Module):
    """Outputs tensors filled with random numbers from distribution depending on value of :code:`distribution`."""
    def __init__(self, distribution: Distributions = 'normal', variance:float | None = None):
        defaults = dict(distribution=distribution, variance=variance)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        distribution = self.defaults['distribution']
        variance = self.get_settings(var.params, 'variance')
        var.update = TensorList(var.params).sample_like(distribution=distribution, variance=variance)
        return var

class Randn(Module):
    """Outputs tensors filled with random numbers from a normal distribution with mean 0 and variance 1."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def step(self, var):
        var.update = [torch.randn_like(p) for p in var.params]
        return var

class Uniform(Module):
    """Outputs tensors filled with random numbers from uniform distribution between :code:`low` and :code:`high`."""
    def __init__(self, low: float, high: float):
        defaults = dict(low=low, high=high)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        low,high = self.get_settings(var.params, 'low','high')
        var.update = [torch.empty_like(t).uniform_(l,h) for t,l,h in zip(var.params, low, high)]
        return var

class GradToNone(Module):
    """Sets :code:`grad` attribute to None on :code:`var`."""
    def __init__(self): super().__init__()
    def step(self, var):
        var.grad = None
        return var

class UpdateToNone(Module):
    """Sets :code:`update` attribute to None on :code:`var`."""
    def __init__(self): super().__init__()
    def step(self, var):
        var.update = None
        return var

class Identity(Module):
    """Identity operator that is argument-insensitive. This also can be used as identity hessian for trust region methods."""
    def __init__(self, *args, **kwargs): super().__init__()
    def step(self, var): return var
    def get_H(self, var):
        n = sum(p.numel() for p in var.params)
        p = var.params[0]
        return ScaledIdentity(shape=(n,n), device=p.device, dtype=p.dtype)

Noop = Identity
"""A placeholder identity operator that is argument-insensitive."""
