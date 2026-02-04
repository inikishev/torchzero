from typing import Any
import torch

from ...core import TensorTransform
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states


def lion_(tensors: TensorList | Any, exp_avg_: TensorList | Any, beta1, beta2,):
    update = exp_avg_.lerp(tensors, 1-beta1).sign_()
    exp_avg_.lerp_(tensors, 1-beta2)
    return update


class Lion(TensorTransform):
    """Lion (EvoLved Sign Momentum) optimizer from https://arxiv.org/abs/2302.06675.

    Args:
        beta1 (float, optional): dampening for momentum. Defaults to 0.9.
        beta2 (float, optional): momentum factor. Defaults to 0.99.
    """

    def __init__(self, beta1: float = 0.9, beta2: float = 0.99):
        defaults = dict(beta1=beta1, beta2=beta2)
        super().__init__(defaults)

        self.add_projected_keys("grad", "exp_avg")

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        beta1, beta2 = unpack_dicts(settings, 'beta1', 'beta2', cls=NumberList)
        exp_avg = unpack_states(states, tensors, 'exp_avg', cls=TensorList)
        return lion_(TensorList(tensors), exp_avg, beta1, beta2)

