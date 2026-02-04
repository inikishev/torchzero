from collections import deque

import torch

from ...core import TensorTransform
from ...utils import TensorList, unpack_dicts,unpack_states

class UnaryLambda(TensorTransform):
    """Applies ``fn`` to input tensors.

    ``fn`` must accept and return a list of tensors.
    """
    def __init__(self, fn):
        defaults = dict(fn=fn)
        super().__init__(defaults=defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        return settings[0]['fn'](tensors)

class UnaryParameterwiseLambda(TensorTransform):
    """Applies ``fn`` to each input tensor.

    ``fn`` must accept and return a tensor.
    """
    def __init__(self, fn):
        defaults = dict(fn=fn)
        super().__init__(defaults=defaults)

    @torch.no_grad
    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        return setting['fn'](tensor)

class CustomUnaryOperation(TensorTransform):
    """Applies ``getattr(tensor, name)`` to each tensor
    """
    def __init__(self, name: str):
        defaults = dict(name=name)
        super().__init__(defaults=defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        return getattr(tensors, settings[0]['name'])()


class Abs(TensorTransform):
    """Returns ``abs(input)``"""
    def __init__(self): super().__init__()
    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        torch._foreach_abs_(tensors)
        return tensors

class Sign(TensorTransform):
    """Returns ``sign(input)``"""
    def __init__(self): super().__init__()
    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        torch._foreach_sign_(tensors)
        return tensors

class Exp(TensorTransform):
    """Returns ``exp(input)``"""
    def __init__(self): super().__init__()
    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        torch._foreach_exp_(tensors)
        return tensors

class Sqrt(TensorTransform):
    """Returns ``sqrt(input)``"""
    def __init__(self): super().__init__()
    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        torch._foreach_sqrt_(tensors)
        return tensors

class Reciprocal(TensorTransform):
    """Returns ``1 / input``"""
    def __init__(self, eps = 0):
        defaults = dict(eps = eps)
        super().__init__(defaults)
    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        eps = [s['eps'] for s in settings]
        if any(e != 0 for e in eps): torch._foreach_add_(tensors, eps)
        torch._foreach_reciprocal_(tensors)
        return tensors

class Negate(TensorTransform):
    """Returns ``- input``"""
    def __init__(self): super().__init__()
    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        torch._foreach_neg_(tensors)
        return tensors


class NanToNum(TensorTransform):
    """Convert ``nan``, ``inf`` and `-`inf`` to numbers.

    Args:
        nan (optional): the value to replace NaNs with. Default is zero.
        posinf (optional): if a Number, the value to replace positive infinity values with.
            If None, positive infinity values are replaced with the greatest finite value
            representable by input's dtype. Default is None.
        neginf (optional): if a Number, the value to replace negative infinity values with.
            If None, negative infinity values are replaced with the lowest finite value
            representable by input's dtype. Default is None.
    """
    def __init__(self, nan=None, posinf=None, neginf=None):
        defaults = dict(nan=nan, posinf=posinf, neginf=neginf)
        super().__init__(defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        nan, posinf, neginf = unpack_dicts(settings, 'nan', 'posinf', 'neginf')
        return [t.nan_to_num_(nan_i, posinf_i, neginf_i) for t, nan_i, posinf_i, neginf_i in zip(tensors, nan, posinf, neginf)]

class Rescale(TensorTransform):
    """Rescales input to ``(min, max)`` range"""
    def __init__(self, min: float, max: float, tensorwise: bool = False, eps:float=1e-8):
        defaults = dict(min=min, max=max, eps=eps, tensorwise=tensorwise)
        super().__init__(defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        min, max = unpack_dicts(settings, 'min','max')
        tensorwise = settings[0]['tensorwise']
        dim = None if tensorwise else 'global'
        return TensorList(tensors).rescale(min=min, max=max, eps=settings[0]['eps'], dim=dim)