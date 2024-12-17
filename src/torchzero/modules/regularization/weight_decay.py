import typing as T
from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule


def l2_regularize_(params: abc.Iterable[torch.Tensor], alpha: float = 1e-2):
    """Adds L2 weight regularization term to the gradients.

    Args:
        params (abc.Iterable[torch.Tensor]): an iterable of Tensors or a single Tensor.
        alpha (float, optional): multiplier to the regularizer. Defaults to 1e-2.
    """
    p = TensorList(params).with_requires_grad()
    p.ensure_grad_()
    p.grad.add_(p, alpha = alpha)

def l1_regularize_(params: abc.Iterable[torch.Tensor], alpha: float = 1e-2):
    """Adds L1 weight regularization term to the gradients.

    Args:
        params (abc.Iterable[torch.Tensor]): an iterable of Tensors or a single Tensor.
        alpha (float, optional): multiplier to the regularizer. Defaults to 1e-2.
    """
    p = TensorList(params).with_requires_grad()
    p.ensure_grad_()
    p.grad.add_(p.sign(), alpha = alpha)

def weight_decay_penalty(params: abc.Iterable[torch.Tensor], alpha: float, ord = 2):
    """Calculate the weight decay penalty term that can be added to the loss.

    Args:
        params (abc.Iterable[torch.Tensor]): an iterable of Tensors or a single Tensor.
        alpha (float): multiplier to the regularizer.
        ord (int, optional): order of the norm. Defaults to 2.
    """
    return TensorList(params).total_vector_norm(ord) * alpha


class WeightDecay(OptimizerModule):
    def __init__(self, alpha: float = 1e-2, ord:T.Literal[1, 2] = 2, make_closure = False):
        """Adds weight decay term (L1 or L2 regularization) to the ascent direction.

        Put this at the end to make it decoupled.

        Args:
            alpha (float, optional): multiplier to the regularizer. Defaults to 1e-2.
            ord (T.Literal[1, 2], optional): Order, 1 and 2 are currently supported (L1 and L2 regularization).
                Defaults to 2.
            make_closure (bool, optional): if True, instead of directly changing ascent direction,
                this creates a new closure that adds the penalty to the update. Defaults to False.
        """
        defaults = dict(alpha = alpha)
        super().__init__(defaults, make_closure=make_closure)
        self.ord = ord

    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()
        alpha = self.get_group_key('alpha')

        if self.ord == 1:
            ascent.add_(params.sign() * alpha)

        elif self.ord == 2:
            ascent.add_(params * alpha)

        else:
            raise NotImplementedError(f'Ord {self.ord} not implemented.')

        return ascent