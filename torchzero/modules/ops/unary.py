from collections import deque

import torch

from ...core import TensorwiseTransform, Target, Transform
from ...utils import TensorList

class UnaryLambda(Transform):
    def __init__(self, fn, target: "Target" = 'update'):
        defaults = dict(fn=fn)
        super().__init__(defaults=defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        return self.settings[params[0]]['fn'](tensors)

class UnaryParameterwiseLambda(TensorwiseTransform):
    def __init__(self, fn, target: "Target" = 'update'):
        defaults = dict(fn=fn)
        super().__init__(uses_grad=False, defaults=defaults, target=target)

    @torch.no_grad
    def transform(self, tensor, param, grad, vars):
        return self.settings[param]['fn'](tensor)

class CustomUnaryOperation(Transform):
    def __init__(self, name: str, target: "Target" = 'update'):
        defaults = dict(name=name)
        super().__init__(defaults=defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        return getattr(tensors, self.settings[params[0]]['name'])()


class Abs(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__({}, uses_grad=False, target=target)
    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        torch._foreach_abs_(tensors)
        return tensors

class Sign(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__({}, uses_grad=False, target=target)
    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        torch._foreach_sign_(tensors)
        return tensors

class Exp(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__({}, uses_grad=False, target=target)
    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        torch._foreach_exp_(tensors)
        return tensors

class Sqrt(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__({}, uses_grad=False, target=target)
    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        torch._foreach_sqrt_(tensors)
        return tensors

class Reciprocal(Transform):
    def __init__(self, eps = 0, target: "Target" = 'update'):
        defaults = dict(eps = eps)
        super().__init__(defaults, uses_grad=False, target=target)
    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        eps = self.get_settings('eps', params=params)
        if any(e != 0 for e in eps): torch._foreach_add_(tensors, eps)
        torch._foreach_reciprocal_(tensors)
        return tensors

class Negate(Transform):
    def __init__(self, target: "Target" = 'update'): super().__init__({}, uses_grad=False, target=target)
    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        torch._foreach_neg_(tensors)
        return tensors


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
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        nan, posinf, neginf = self.get_settings('nan', 'posinf', 'neginf', params=params)
        return [t.nan_to_num_(nan_i, posinf_i, neginf_i) for t, nan_i, posinf_i, neginf_i in zip(tensors, nan, posinf, neginf)]

class Rescale(Transform):
    """rescale update to (min, max) range"""
    def __init__(self, min: float, max: float, tensorwise: bool = False, eps:float=1e-8, target: "Target" = 'update'):
        defaults = dict(min=min, max=max, eps=eps, tensorwise=tensorwise)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        min,max = self.get_settings('min','max', params=params)
        tensorwise = self.settings[params[0]]['tensorwise']
        dim = None if tensorwise else 'global'
        return TensorList(tensors).rescale(min=min, max=max, eps=self.settings[params[0]]['eps'], dim=dim)