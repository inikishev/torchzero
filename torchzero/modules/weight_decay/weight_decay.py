from collections.abc import Iterable, Sequence
from typing import Literal

import torch

from ...core import Module,  TensorTransform
from ...utils import NumberList, TensorList, as_tensorlist, unpack_dicts, unpack_states, Metrics


@torch.no_grad
def weight_decay_(
    grad_: TensorList,
    params: TensorList,
    weight_decay: float | NumberList,
    ord: int = 2
):
    """modifies in-place and returns ``grad_``."""
    if ord == 1: return grad_.add_(params.sign().mul_(weight_decay))
    if ord == 2: return grad_.add_(params.mul(weight_decay))
    if ord - 1 % 2 != 0: return grad_.add_(params.pow(ord-1).mul_(weight_decay))
    return grad_.add_(params.pow(ord-1).copysign_(params).mul_(weight_decay))


class WeightDecay(TensorTransform):
    """Weight decay.

    Args:
        weight_decay (float): weight decay scale.
        ord (int, optional): order of the penalty, e.g. 1 for L1 and 2 for L2. Defaults to 2.
        target (Target, optional): what to set on var. Defaults to 'update'.

    ### Examples:

    Adam with non-decoupled weight decay
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.WeightDecay(1e-3),
        tz.m.Adam(),
        tz.m.LR(1e-3)
    )
    ```

    Adam with decoupled weight decay that still scales with learning rate
    ```python

    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Adam(),
        tz.m.WeightDecay(1e-3),
        tz.m.LR(1e-3)
    )
    ```

    Adam with fully decoupled weight decay that doesn't scale with learning rate
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Adam(),
        tz.m.LR(1e-3),
        tz.m.WeightDecay(1e-6)
    )
    ```

    """
    def __init__(self, weight_decay: float, ord: int = 2):

        defaults = dict(weight_decay=weight_decay, ord=ord)
        super().__init__(defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        weight_decay = NumberList(s['weight_decay'] for s in settings)
        ord = settings[0]['ord']

        return weight_decay_(as_tensorlist(tensors), as_tensorlist(params), weight_decay, ord)

class RelativeWeightDecay(TensorTransform):
    """Weight decay relative to the mean absolute value of update, gradient or parameters depending on value of ``norm_input`` argument.

    Args:
        weight_decay (float): relative weight decay scale.
        ord (int, optional): order of the penalty, e.g. 1 for L1 and 2 for L2. Defaults to 2.
        norm_input (str, optional):
            determines what should weight decay be relative to. "update", "grad" or "params".
            Defaults to "update".
        metric (Ords, optional):
            metric (norm, etc) that weight decay should be relative to.
            defaults to 'mad' (mean absolute deviation).
        target (Target, optional): what to set on var. Defaults to 'update'.

    ### Examples:

    Adam with non-decoupled relative weight decay
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.RelativeWeightDecay(1e-1),
        tz.m.Adam(),
        tz.m.LR(1e-3)
    )
    ```

    Adam with decoupled relative weight decay
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Adam(),
        tz.m.RelativeWeightDecay(1e-1),
        tz.m.LR(1e-3)
    )
    ```
    """
    def __init__(
        self,
        weight_decay: float = 0.1,
        ord: int  = 2,
        norm_input: Literal["update", "grad", "params"] = "update",
        metric: Metrics = 'mad',
    ):
        defaults = dict(weight_decay=weight_decay, ord=ord, norm_input=norm_input, metric=metric)
        super().__init__(defaults, uses_grad=norm_input == 'grad')

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        weight_decay = NumberList(s['weight_decay'] for s in settings)

        ord = settings[0]['ord']
        norm_input = settings[0]['norm_input']
        metric = settings[0]['metric']

        if norm_input == 'update': src = TensorList(tensors)
        elif norm_input == 'grad':
            assert grads is not None
            src = TensorList(grads)
        elif norm_input == 'params':
            src = TensorList(params)
        else:
            raise ValueError(norm_input)

        norm = src.global_metric(metric)
        return weight_decay_(as_tensorlist(tensors), as_tensorlist(params), weight_decay * norm, ord)


@torch.no_grad
def decay_weights_(params: Iterable[torch.Tensor], weight_decay: float | NumberList, ord:int=2):
    """directly decays weights in-place"""
    params = TensorList(params)
    weight_decay_(params, params, -weight_decay, ord)

class DirectWeightDecay(Module):
    """Directly applies weight decay to parameters.

    Args:
        weight_decay (float): weight decay scale.
        ord (int, optional): order of the penalty, e.g. 1 for L1 and 2 for L2. Defaults to 2.
    """
    def __init__(self, weight_decay: float, ord: int = 2,):
        defaults = dict(weight_decay=weight_decay, ord=ord)
        super().__init__(defaults)

    @torch.no_grad
    def apply(self, objective):
        weight_decay = self.get_settings(objective.params, 'weight_decay', cls=NumberList)
        ord = self.defaults['ord']

        decay_weights_(objective.params, weight_decay, ord)
        return objective