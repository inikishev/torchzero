"""
⟂Grad (read “ortho-grad”) was proposed in https://arxiv.org/abs/2501.04697.

"""
import logging
from collections.abc import Iterable, Sequence

import numpy as np
import torch

from ... import tl
from ...core import ClosureType, OptimizationState, OptimizerModule
from ...python_tools import _ScalarLoss


def orthograd_(params: Iterable[torch.Tensor], eps: float = 1e-30):
    """Applies ⟂Grad - projects gradient of an iterable of parameters to be orthogonal to the weights.

    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to apply ⟂Grad to.
        eps (float, optional): epsilon added to the denominator for numerical stability (default: 1e-30)

    reference
        https://arxiv.org/abs/2501.04697
    """
    if not isinstance(params, tl.TensorList): params = tl.TensorList(params)
    params = params.with_grad()
    grad = params.grad
    grad -= (((params*grad).total_sum())/(params*params).total_sum() + eps) * params

class OrthoGrad(OptimizerModule):
    """⟂Grad - projects gradient of an iterable of parameters to be orthogonal to the weights.

    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to apply ⟂Grad to.
        eps (float, optional): epsilon added to the denominator for numerical stability (default: 1e-30)

    reference
        https://arxiv.org/abs/2501.04697
    """
    def __init__(self, eps: float = 1e-30):
        super().__init__({})
        self.eps = eps

    def _update(self, state, ascent):
        params = self.get_params()
        ascent -= (((params*ascent).total_sum())/(params*params).total_sum() + self.eps) * params
        return ascent

