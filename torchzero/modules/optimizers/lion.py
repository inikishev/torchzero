import torch

from ...core import Module, Target, Transform
from ...utils import NumberList, TensorList


def lion_(tensors: TensorList, exp_avg_: TensorList, beta1, beta2,):
    """
    Lion update rule.

    Returns new tensors.
    """
    update = exp_avg_.lerp(tensors, 1-beta1).sign_()
    exp_avg_.lerp_(tensors, 1-beta2)
    return update


class Lion(Transform):
    """Lion (EvoLved Sign Momentum) optimizer from https://arxiv.org/abs/2302.06675.

    Args:
        beta1 (float, optional): dampening for momentum. Defaults to 0.9.
        beta2 (float, optional): momentum factor. Defaults to 0.99.
    """

    def __init__(self, beta1: float = 0.9, beta2: float = 0.99):
        defaults = dict(beta1=beta1, beta2=beta2)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        beta1, beta2 = self.get_settings('beta1', 'beta2', params = params, cls=NumberList)
        exp_avg = self.get_state('ema', params=params, cls=TensorList)
        return lion_(TensorList(tensors),exp_avg,beta1,beta2)

