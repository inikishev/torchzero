from collections.abc import Iterable, Sequence
from typing import Literal

import torch

from ...core import Module, Target, Transform
from ...utils import NumberList, TensorList, as_tensorlist, unpack_dicts, unpack_states


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
        weight_decay = NumberList(s['weight_decay'] for s in settings)
        ord = settings[0]['ord']

        return weight_decay_(as_tensorlist(tensors), as_tensorlist(params), weight_decay, ord)

class NormalizedWeightDecay(Transform):
    def __init__(
        self,
        weight_decay: float = 0.1,
        ord: int = 2,
        norm_input: Literal["update", "grad", "params"] = "update",
        target: Target = "update",
    ):
        defaults = dict(weight_decay=weight_decay, ord=ord, norm_input=norm_input)
        super().__init__(defaults, uses_grad=norm_input == 'grad', target=target)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        weight_decay = NumberList(s['weight_decay'] for s in settings)

        ord = settings[0]['ord']
        norm_input = settings[0]['norm_input']

        if norm_input == 'update': src = TensorList(tensors)
        elif norm_input == 'grad':
            assert grads is not None
            src = TensorList(grads)
        elif norm_input == 'params':
            src = TensorList(params)
        else:
            raise ValueError(norm_input)

        norm = src.global_vector_norm(ord)

        return weight_decay_(as_tensorlist(tensors), as_tensorlist(params), weight_decay * norm, ord)


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
        weight_decay = self.get_settings(var.params, 'weight_decay', cls=NumberList)
        ord = self.settings[var.params[0]]['ord']

        decay_weights_(var.params, weight_decay, ord)
        return var