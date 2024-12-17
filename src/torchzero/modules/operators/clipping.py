import typing
from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule
from .normalization import _normalize_grad

def clip_grad_value_(params: abc.Iterable[torch.Tensor], value:float):
    """Clip the gradients of an iterable of parameters at specified value.

    Args:
        params (abc.Iterable[torch.Tensor]): an iterable of Tensors or a single Tensor that will have gradients clipped.
        value (float, optional): maximum allowed magnitude of the gradients.
            The gradients are clipped in the range `[-clip_value, clip_value]`
    """
    TensorList(params).get_existing_grads().clamp_(-value, value)

class ClipValue(OptimizerModule):
    def __init__(self, value: float):
        """Clip the update at specified value.

        Args:
        value (float, optional): maximum allowed magnitude of the gradients.
            The gradients are clipped in the range `[-clip_value, clip_value]`
        """
        defaults = dict(value = value)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent):
        value = self.get_group_key('value')
        ascent.clamp_(-value, value)
        return ascent

def clip_grad_norm_(
    params: abc.Iterable[torch.Tensor],
    max_norm: float,
    ord: float = 2,
    mode: typing.Literal["global", "param", "channel"] = "param",
):
    """Clip the gradient norm of an iterable of parameters.

    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to clip the norm of.
        max_norm (float, optional): norm value to clip to.
        ord (float, optional): order of the norm. Defaults to 2.
        mode (str, optional): what to calculate the norm over.

            - "global": calculates and clips the norm of the entire gradient, as if it was a single vector.

            - "param": calculates and clips each param's gradient norm (default).

            - "channel": calculate and clip the norm of gradient of each channel of each param.

    Example:
        >>> clip_grad_norm_(model.parameters())
    """
    _normalize_grad(
        (p.grad for p in params if p.grad is not None),
        norm_value = max_norm,
        min = max_norm,
        ord = ord,
        mode = mode,
    )

class ClipNorm(OptimizerModule):
    def __init__(self, max_norm: float, ord:float=2, mode: typing.Literal["global", "param", "channel"] = "param",):
        """Clip the gradient norm of an iterable of parameters.

        Args:
            max_norm (float, optional): norm value to clip to.
            ord (float, optional): order of the norm. Defaults to 2.
            mode (str, optional): what to calculate the norm over.

                - "global": calculates and clips the norm of the entire gradient, as if it was a single vector.

                - "param": calculates and clips each param's gradient norm (default).

                - "channel": calculate and clip the norm of gradient of each channel of each param.
        """
        super().__init__({})
        self.max_norm = max_norm
        self.ord = ord
        self.mode: typing.Literal["global", "param", "channel"] = mode

    @torch.no_grad
    def _update(self, state, ascent):
        _normalize_grad(
            ascent,
            norm_value = self.max_norm,
            min = self.max_norm,
            ord = self.ord,
            mode = self.mode,
        )
