import torch

from ...core import Target, Transform
from ...utils import TensorList, unpack_states

class AccumulateSum(Transform):
    """Accumulates sum of all past updates.

    Args:
        decay (float, optional): decays the accumulator. Defaults to 0.
        target (Target, optional): target. Defaults to 'update'.
    """
    def __init__(self, decay: float = 0, target: Target = 'update',):
        defaults = dict(decay=decay)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        sum = unpack_states(states, tensors, 'sum', cls=TensorList)
        decay = [1-s['decay'] for s in settings]
        return sum.add_(tensors).lazy_mul(decay, clone=True)

class AccumulateMean(Transform):
    """Accumulates mean of all past updates.

    Args:
        decay (float, optional): decays the accumulator. Defaults to 0.
        target (Target, optional): target. Defaults to 'update'.
    """
    def __init__(self, decay: float = 0, target: Target = 'update',):
        defaults = dict(decay=decay)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1
        mean = unpack_states(states, tensors, 'mean', cls=TensorList)
        decay = [1-s['decay'] for s in settings]
        return mean.add_(tensors).lazy_mul(decay, clone=True).div_(step)

class AccumulateProduct(Transform):
    """Accumulates product of all past updates.

    Args:
        decay (float, optional): decays the accumulator. Defaults to 0.
        target (Target, optional): target. Defaults to 'update'.
    """
    def __init__(self, decay: float = 0, target: Target = 'update',):
        defaults = dict(decay=decay)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        prod = unpack_states(states, tensors, 'prod', cls=TensorList)
        decay = [1-s['decay'] for s in settings]
        return prod.mul_(tensors).lazy_mul(decay, clone=True)

class AccumulateMaximum(Transform):
    """Accumulates maximum of all past updates.

    Args:
        decay (float, optional): decays the accumulator. Defaults to 0.
        target (Target, optional): target. Defaults to 'update'.
    """
    def __init__(self, decay: float = 0, target: Target = 'update',):
        defaults = dict(decay=decay)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        maximum = unpack_states(states, tensors, 'maximum', cls=TensorList)
        decay = [1-s['decay'] for s in settings]
        return maximum.maximum_(tensors).lazy_mul(decay, clone=True)

class AccumulateMinimum(Transform):
    """Accumulates minimum of all past updates.

    Args:
        decay (float, optional): decays the accumulator. Defaults to 0.
        target (Target, optional): target. Defaults to 'update'.
    """
    def __init__(self, decay: float = 0, target: Target = 'update',):
        defaults = dict(decay=decay)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        minimum = unpack_states(states, tensors, 'minimum', cls=TensorList)
        decay = [1-s['decay'] for s in settings]
        return minimum.minimum_(tensors).lazy_mul(decay, clone=True)

