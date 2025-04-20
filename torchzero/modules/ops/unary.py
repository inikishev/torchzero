from collections import deque

import torch

from ...core import ParameterwiseTransform, Target, Transform
from ...utils import TensorList

class UnaryLambda(Transform):
    def __init__(self, fn, target: "Target" = 'update'):
        defaults = dict(fn=fn)
        super().__init__(defaults=defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        return self.defaults['fn'](target)

class UnaryParameterwiseLambda(ParameterwiseTransform):
    def __init__(self, fn, target: "Target" = 'update'):
        defaults = dict(fn=fn)
        super().__init__(requires_grad=False, defaults=defaults, target=target)

    @torch.no_grad
    def transform(self, target, param, grad, vars):
        return self.settings[param]['fn'](target)

class CustomUnaryOperation(Transform):
    def __init__(self, name: str, target: "Target" = 'update'):
        defaults = dict(name=name)
        super().__init__(defaults=defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        return getattr(target, self.defaults['name'])()


class Abs(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__(target=target)
    @torch.no_grad
    def transform(self, target, vars):
        torch._foreach_abs_(target)
        return target

class Sign(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__(target=target)
    @torch.no_grad
    def transform(self, target, vars):
        torch._foreach_sign_(target)
        return target

class Exp(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__(target=target)
    @torch.no_grad
    def transform(self, target, vars):
        torch._foreach_exp_(target)
        return target

class Sqrt(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__(target=target)
    @torch.no_grad
    def transform(self, target, vars):
        torch._foreach_sqrt_(target)
        return target

class Reciprocal(Transform):
    def __init__(self, eps = 0, target: "Target" = 'update'):
        defaults = dict(eps = eps)
        super().__init__(defaults, target=target)
    @torch.no_grad
    def transform(self, target, vars):
        eps = self.get_settings('eps', params=vars)
        if any(e != 0 for e in eps): torch._foreach_add_(target, eps)
        torch._foreach_reciprocal_(target)
        return target

class Negate(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__(target=target)
    @torch.no_grad
    def transform(self, target, vars):
        torch._foreach_neg_(target)
        return target


class NanToNum(Transform):
    """Convert `nan`, `inf` and `-inf` to numbers.

    Args:
        nan (optional): the value to replace NaNs with. Default is zero.
        posinf (optional): if a Number, the value to replace positive infinity values with.
            If None, positive infinity values are replaced with the greatest finite value
            representable by input's dtype. Default is None.
        neginf (optional): if a Number, the value to replace negative infinity values with.
            If None, negative infinity values are replaced with the lowest finite value
            representable by input's dtype. Default is None.
    """
    def __init__(self, nan=None, posinf=None, neginf=None, target: "Target" = 'update'):
        defaults = dict(nan=nan, posinf=posinf, neginf=neginf)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        nan, posinf, neginf = self.get_settings('nan', 'posinf', 'neginf', params=vars)
        return [t.nan_to_num_(nan_i, posinf_i, neginf_i) for t, nan_i, posinf_i, neginf_i in zip(target, nan, posinf, neginf)]

class Rescale(Transform):
    """rescale update to (min, max) range"""
    def __init__(self, min: float, max: float, tensorwise: bool = False, eps:float=1e-8, target: "Target" = 'update'):
        defaults = dict(min=min, max=max, eps=eps, tensorwise=tensorwise)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        min,max = self.get_settings('min','max', params=vars)
        tensorwise = self.defaults['tensorwise']
        dim = None if tensorwise else 'global'
        return TensorList(target).rescale(min=min, max=max, eps=self.defaults['eps'], dim=dim)