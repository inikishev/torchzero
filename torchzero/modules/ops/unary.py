from collections import deque

import torch

from ...core import TensorwiseTransform, Target, Transform
from ...utils import TensorList, unpack_dicts,unpack_states

class UnaryLambda(Transform):
    """Applies :code:`fn` to input tensors.

    :code:`fn` must accept and return a list of tensors.
    """
    def __init__(self, fn, target: "Target" = 'update'):
        defaults = dict(fn=fn)
        super().__init__(defaults=defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        return settings[0]['fn'](tensors)

class UnaryParameterwiseLambda(TensorwiseTransform):
    """Applies :code:`fn` to each input tensor.

    :code:`fn` must accept and return a tensor.
    """
    def __init__(self, fn, target: "Target" = 'update'):
        defaults = dict(fn=fn)
        super().__init__(uses_grad=False, defaults=defaults, target=target)

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, loss, state, setting):
        return setting['fn'](tensor)

class CustomUnaryOperation(Transform):
    """Applies :code:`getattr(tensor, name)` to each tensor
    """
    def __init__(self, name: str, target: "Target" = 'update'):
        defaults = dict(name=name)
        super().__init__(defaults=defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        return getattr(tensors, settings[0]['name'])()


class Abs(Transform):
    """Returns :code:`abs(input)`"""
    def __init__(self, target: "Target" = 'update'): super().__init__({}, uses_grad=False, target=target)
    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        torch._foreach_abs_(tensors)
        return tensors

class Sign(Transform):
    """Returns :code:`sign(input)`"""
    def __init__(self, target: "Target" = 'update'): super().__init__({}, uses_grad=False, target=target)
    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        torch._foreach_sign_(tensors)
        return tensors

class Exp(Transform):
    """Returns :code:`exp(input)`"""
    def __init__(self, target: "Target" = 'update'): super().__init__({}, uses_grad=False, target=target)
    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        torch._foreach_exp_(tensors)
        return tensors

class Sqrt(Transform):
    """Returns :code:`sqrt(input)`"""
    def __init__(self, target: "Target" = 'update'): super().__init__({}, uses_grad=False, target=target)
    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        torch._foreach_sqrt_(tensors)
        return tensors

class Reciprocal(Transform):
    """Returns :code:`1 / input`"""
    def __init__(self, eps = 0, target: "Target" = 'update'):
        defaults = dict(eps = eps)
        super().__init__(defaults, uses_grad=False, target=target)
    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        eps = [s['eps'] for s in settings]
        if any(e != 0 for e in eps): torch._foreach_add_(tensors, eps)
        torch._foreach_reciprocal_(tensors)
        return tensors

class Negate(Transform):
    """Returns :code:`- input`"""
    def __init__(self, target: "Target" = 'update'): super().__init__({}, uses_grad=False, target=target)
    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
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
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        nan, posinf, neginf = unpack_dicts(settings, 'nan', 'posinf', 'neginf')
        return [t.nan_to_num_(nan_i, posinf_i, neginf_i) for t, nan_i, posinf_i, neginf_i in zip(tensors, nan, posinf, neginf)]

class Rescale(Transform):
    """Rescales input to :code`(min, max)` range"""
    def __init__(self, min: float, max: float, tensorwise: bool = False, eps:float=1e-8, target: "Target" = 'update'):
        defaults = dict(min=min, max=max, eps=eps, tensorwise=tensorwise)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        min, max = unpack_dicts(settings, 'min','max')
        tensorwise = settings[0]['tensorwise']
        dim = None if tensorwise else 'global'
        return TensorList(tensors).rescale(min=min, max=max, eps=settings[0]['eps'], dim=dim)