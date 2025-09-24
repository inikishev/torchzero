import math

import torch

from ...core import TensorTransform
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states

# i've verified, it is identical to official
# https://github.com/txping/AEGD/blob/master/aegd.py
# TODO: add a test
def aegd_(f: torch.Tensor | float, g: TensorList, r_: TensorList, c:float|NumberList=1, eta:float|NumberList=0.1) -> TensorList:
    v = g / (2 * (f + c)**0.5)
    r_ /= 1 + (v ** 2).mul_(2*eta) # update energy
    return 2*eta * r_*v # pyright:ignore[reportReturnType]

class AEGD(TensorTransform):
    """AEGD (Adaptive gradient descent with energy) from https://arxiv.org/abs/2010.05109#page=10.26.

    Note:
        AEGD has a learning rate hyperparameter that can't really be removed from the update rule.
        To avoid compounding learning rate mofications, remove the ``tz.m.LR`` module if you had it.

    Args:
        lr (float, optional): learning rate (default: 0.1)
        c (float, optional): term added to the original objective function (default: 1)

    Reference:
        [Liu, Hailiang, and Xuping Tian. "AEGD: Adaptive gradient descent with energy." arXiv preprint arXiv:2010.05109 (2020).](https://arxiv.org/pdf/2010.05109)
    """
    def __init__(
        self,
        lr: float = 0.1,
        c: float = 1,
    ):
        defaults = dict(c=c, lr=lr)
        super().__init__(defaults, uses_loss=True)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        assert loss is not None
        tensors = TensorList(tensors)

        c, lr = unpack_dicts(settings, 'c', 'lr', cls=NumberList)
        r = unpack_states(states, tensors, 'r', init=lambda t: torch.full_like(t, float(loss + c[0])**0.5), cls=TensorList)

        update = aegd_(
            f=loss,
            g=tensors,
            r_=r,
            c=c,
            eta=lr,
        )

        return update