from collections.abc import Iterable, Sequence

import torch

from ...core import Module, Target, Transform
from ...utils import NumberList, TensorList, as_tensorlist

@torch.no_grad
def weight_decay_(
    grad_: TensorList,
    params: TensorList,
    weight_decay: float | NumberList,
    ord: int = 2
):
    """returns `grad_`."""
    if ord == 1: return grad_.add_(params.sign().mul_(weight_decay))
    if ord == 2: return grad_.add_(params.mul(weight_decay))
    if ord - 1 % 2 != 0: return grad_.add_(params.pow(ord-1).mul_(weight_decay))
    return grad_.add_(params.pow(ord-1).copysign_(params).mul_(weight_decay))


class WeightDecay(Transform):
    def __init__(self, weight_decay: float, ord: int = 2, target: Target = 'update'):
        defaults = dict(weight_decay=weight_decay, ord=ord)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        weight_decay = self.get_settings('weight_decay', params=params, cls=NumberList)
        ord = self.defaults['ord']

        return weight_decay_(as_tensorlist(target), as_tensorlist(params), weight_decay, ord)

@torch.no_grad
def decay_weights_(params: Iterable[torch.Tensor], weight_decay: float | NumberList, ord:int=2):
    """directly decays weights in-place"""
    params = TensorList(params)
    weight_decay_(params, params, -weight_decay, ord)

class DirectWeightDecay(Module):
    """directly decays weights in-place"""
    def __init__(self, weight_decay: float, ord: int = 2,):
        defaults = dict(weight_decay=weight_decay, ord=ord)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, vars):
        weight_decay = self.get_settings('weight_decay', params=vars.params, cls=NumberList)
        ord = self.defaults['ord']

        decay_weights_(vars.params, weight_decay, ord)
        return vars