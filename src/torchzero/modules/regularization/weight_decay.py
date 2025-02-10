from typing import Literal
from collections.abc import Iterable

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule, _Targets


def l2_regularize_(params: Iterable[torch.Tensor], alpha: float = 1e-2):
    """Adds L2 weight regularization term to the gradients in-place.

    Args:
        params (Iterable[torch.Tensor]): an iterable of Tensors or a single Tensor.
        alpha (float, optional): multiplier to the regularizer. Defaults to 1e-2.
    """
    p = TensorList(params).with_requires_grad()
    p.ensure_grad_()
    p.grad.add_(p, alpha = alpha)

def l1_regularize_(params: Iterable[torch.Tensor], alpha: float = 1e-2):
    """Adds L1 weight regularization term to the gradients in-place.

    Args:
        params (Iterable[torch.Tensor]): an iterable of Tensors or a single Tensor.
        alpha (float, optional): multiplier to the regularizer. Defaults to 1e-2.
    """
    p = TensorList(params).with_requires_grad()
    p.ensure_grad_()
    p.grad.add_(p.sign(), alpha = alpha)

def weight_decay_penalty(params: Iterable[torch.Tensor], alpha: float = 1e-2, ord:float = 2):
    """Calculate the weight decay penalty term that can be added to the loss.

    Args:
        params (Iterable[torch.Tensor]): an iterable of Tensors or a single Tensor.
        alpha (float): multiplier to the regularizer.
        ord (int, optional): order of the norm. Defaults to 2.
    """
    return TensorList(params).norm(ord) * alpha

def decay_weights_(params: Iterable[torch.Tensor], alpha: float = 1e-2, ord:Literal[1, 2] = 2):
    """Apply weight decay directly to parameters in-place.

    Args:
        params (Iterable[torch.Tensor]): an iterable of Tensors or a single Tensor to decay.
        alpha (float): by how much to decay parameters (default: 1e-2)
        ord (float, optional):
            order of the penalty, 1 and 2 are currently supported (L1 and L2 regularization) (default: 2)
    """
    params = TensorList(params)
    if ord == 2: params.mul_(1-alpha)
    elif ord == 1: params.sub_(params.sign().mul_(alpha))
    else: raise NotImplementedError(f'order {ord} is not supported')


class WeightDecay(OptimizerModule):
    """Adds weight decay term (L1 or L2 regularization) to the ascent direction.

    Put this at the end to make it decoupled.

    Args:
        alpha (float, optional): multiplier to the regularizer (default: 1e-2)
        ord (Literal[1, 2], optional):
            order of the penalty, 1 and 2 are currently supported (L1 and L2 regularization).
            Defaults to 2.
        target (str, optional):
            determines what this module updates.

            "ascent" - it updates the ascent

            "grad" - it updates the gradient (and sets `.grad` attributes to updated gradient).

            "closure" - it makes a new closure that sets the updated ascent to the .`grad` attributes.
    """
    def __init__(self, alpha: float = 1e-2, ord:Literal[1, 2] = 2, target: _Targets = "ascent"):
        defaults = dict(alpha = alpha)
        super().__init__(defaults, target = target)
        self.ord = ord

    @torch.no_grad
    def _update(self, vars, ascent):
        params = self.get_params()
        alpha = self.get_group_key('alpha')

        if any(i != 0 for i in alpha):

            if self.ord == 1: ascent.add_(params.sign() * alpha)
            elif self.ord == 2: ascent.add_(params * alpha)
            else: raise NotImplementedError(f'weight descent of order {self.ord} not implemented.')

        return ascent