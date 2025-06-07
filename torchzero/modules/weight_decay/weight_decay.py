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
    def apply(self, tensors, params, grads, loss, states, settings):
        weight_decay = self.get_settings('weight_decay', params=params, cls=NumberList)
        ord = self.settings[params[0]]['ord']

        return weight_decay_(as_tensorlist(tensors), as_tensorlist(params), weight_decay, ord)

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
    def step(self, var):
        weight_decay = self.get_settings('weight_decay', params=var.params, cls=NumberList)
        ord = self.settings[var.params[0]]['ord']

        decay_weights_(var.params, weight_decay, ord)
        return var