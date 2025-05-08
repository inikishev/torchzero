from collections import deque
from operator import itemgetter
from typing import Literal

import torch

from ...core import Target, Transform
from ...utils import TensorList, NumberList

class AccumulateSum(Transform):
    def __init__(self, decay: float = 0, target: Target = 'update',):
        defaults = dict(decay=decay)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        sum = self.get_state('sum', params=params, cls=TensorList)
        decay = self.get_settings('decay', params=params, cls=NumberList)
        return sum.add_(target).lazy_mul(1-decay, clone=True)

class AccumulateMean(Transform):
    def __init__(self, decay: float = 0, target: Target = 'update',):
        defaults = dict(decay=decay)
        super().__init__(defaults, uses_grad=False, target=target)
        self.global_state['step'] = 0

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        self.global_state['step'] = self.global_state.get('step', 0) + 1
        mean = self.get_state('mean', params=params, cls=TensorList)
        decay = self.get_settings('decay', params=params, cls=NumberList)
        return mean.add_(target).lazy_mul(1-decay, clone=True).div_(self.global_state['step'])

class AccumulateProduct(Transform):
    def __init__(self, decay: float = 0, target: Target = 'update',):
        defaults = dict(decay=decay)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        prod = self.get_state('prod', params=params, cls=TensorList)
        decay = self.get_settings('decay', params=params, cls=NumberList)
        return prod.mul_(target).lazy_mul(1-decay, clone=True)

class AccumulateMaximum(Transform):
    def __init__(self, decay: float = 0, target: Target = 'update',):
        defaults = dict(decay=decay)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        maximum = self.get_state('maximum', params=params, cls=TensorList)
        decay = self.get_settings('decay', params=params, cls=NumberList)
        return maximum.maximum_(target).lazy_mul(1-decay, clone=True)

class AccumulateMinimum(Transform):
    def __init__(self, decay: float = 0, target: Target = 'update',):
        defaults = dict(decay=decay)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        minimum = self.get_state('minimum', params=params, cls=TensorList)
        decay = self.get_settings('decay', params=params, cls=NumberList)
        return minimum.minimum_(target).lazy_mul(1-decay, clone=True)

