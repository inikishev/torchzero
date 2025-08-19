from operator import itemgetter
import math
import warnings
from collections.abc import Iterable, Sequence
from typing import Literal

import torch

from ...core import Target, Transform
from ...utils import as_tensorlist

def orthograd_(params: Iterable[torch.Tensor], eps: float = 1e-30):
    """Applies ⟂Grad - projects gradient of an iterable of parameters to be orthogonal to the weights.

    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to apply ⟂Grad to.
        eps (float, optional): epsilon added to the denominator for numerical stability (default: 1e-30)

    reference
        https://arxiv.org/abs/2501.04697
    """
    params = as_tensorlist(params).with_grad()
    grad = params.grad
    grad -= (params.dot(grad)/(params.dot(params) + eps)) * params


class OrthoGrad(Transform):
    """Applies ⟂Grad - projects gradient of an iterable of parameters to be orthogonal to the weights.

    Args:
        eps (float, optional): epsilon added to the denominator for numerical stability (default: 1e-30)
        renormalize (bool, optional): whether to graft projected gradient to original gradient norm. Defaults to True.
        target (Target, optional): what to set on var. Defaults to 'update'.
    """
    def __init__(self, eps: float = 1e-8, renormalize=True, target: Target = 'update'):
        defaults = dict(eps=eps, renormalize=renormalize)
        super().__init__(defaults, uses_grad=False, target=target)

    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        eps = settings[0]['eps']
        renormalize = settings[0]['renormalize']

        params = as_tensorlist(params)
        target = as_tensorlist(tensors)

        scale = params.dot(target)/(params.dot(params) + eps)
        if renormalize:
            norm = target.global_vector_norm()
            target -= params * scale
            target *= (norm / target.global_vector_norm())
            return target

        target -= params * scale
        return target