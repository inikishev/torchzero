from collections import deque
from operator import itemgetter
from typing import Literal

import torch

from ...core import ParameterwiseTransform, Target, Transform
from ...utils import NumberList, TensorList


def cautious_(
    tensors_: TensorList,
    grad: TensorList,
    normalize: bool,
    eps: float,
    mode: Literal['zero', 'grad', 'backtrack']
):
    # mask will be > 0 for parameters where both signs are the same
    mask = (tensors_ * grad) > 0
    if mode in ('zero', 'grad'):
        if normalize and mode == 'zero':
            fmask = mask.to(tensors_[0].dtype)
            fmask /= fmask.global_mean().clip(min=eps) # type:ignore
        else:
            fmask = mask

        tensors_ *= fmask

        if mode == 'grad':
            tensors_ += grad * mask.logical_not_()

        return tensors_

    # mode = 'backtrack'
    tensors_ -= tensors_.mul(2).mul_(mask.logical_not_())
    return tensors_

class Cautious(Transform):
    """Negates update for parameters where update and gradient sign is inconsistent.
    Optionally normalizes the update by the number of parameters that are not masked.
    This is meant to be used after any momentum-based modules.

    Args:
        normalize (bool, optional):
            renormalize update after masking.
            only has effect when mode is 'zero'. Defaults to False.
        eps (float, optional): epsilon for normalization. Defaults to 1e-6.
        mode (str, optional):
            what to do with updates with inconsistent signs.

            "zero" - set them to zero (as in paper)

            "grad" - set them to the gradient

            "backtrack" - negate them (same as using update magnitude and gradient sign)

    reference
        *Cautious Optimizers: Improving Training with One Line of Code.
        Kaizhao Liang, Lizhang Chen, Bo Liu, Qiang Liu*
    """

    def __init__(
        self,
        normalize=False,
        eps=1e-6,
        mode: Literal["zero", "grad", "backtrack"] = "zero",
        target: Target = "update",
    ):
        defaults = dict(normalize=normalize, eps=eps, mode=mode)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        grad = vars.get_grad()
        mode, normalize, eps = itemgetter('mode', 'normalize', 'eps')(self.defaults)
        return cautious_(TensorList(target), TensorList(grad), normalize=normalize, eps=eps, mode=mode)

class UpdateGradientSignConsistency(Transform):
    """1 where signs match 0 otherwise"""
    def __init__(self, normalize = False, eps=1e-6, target: Target = 'update'):
        defaults = dict(normalize=normalize, eps=eps)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def transform(self, target, vars):
        grad = vars.get_grad()
        normalize, eps = itemgetter('normalize', 'eps')(self.defaults)

        mask = (TensorList(target).mul_(grad)).gt_(0)
        if normalize: mask = mask / mask.global_mean().clip(min = eps) # pyright: ignore[reportOperatorIssue]

        return mask
