from collections import abc
import typing
import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule


def _normalize_grad(
    grads: abc.Iterable[torch.Tensor],
    norm_value: float = 1,
    ord: float = 2,
    min: float = 0,
    mode: typing.Literal["global", "param", "channel"] = "param",
    min_numel=2,
):
    if mode in ('param', 'channel'):
        for grad in grads:
            if grad.numel() >= min_numel:
                if mode == 'channel':
                    norm = torch.linalg.vector_norm(grad, ord, dim=tuple(range(1, grad.ndim)), keepdim=True) # pylint:disable=not-callable
                    norm[norm<=min] = 1
                    grad /= norm / norm_value
                else: # mode = 'param'
                    norm = torch.linalg.vector_norm(grad, ord) # pylint:disable=not-callable
                    if norm > min:
                        grad /= norm / norm_value
    else:
        if not isinstance(grads, TensorList): grads = TensorList(grads)
        norm = grads.total_vector_norm(ord)
        if norm > min:
            grads /= norm / norm_value

@torch.no_grad
def normalize_grad_(
    params: abc.Iterable[torch.Tensor],
    norm_value: float = 1,
    ord: float = 2,
    min: float = 0,
    mode: typing.Literal["global", "param", "channel"] = "global",
    min_numel=2,
):
    """Normalizes gradients of an iterable of parameters.

    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to normalize.
        norm_value (float, optional): value to normalize to. Defaults to 1.
        ord (float, optional): order of the norm. Defaults to 2.
        min (float, optional):
            won't normalize when gradient is below this norm, you can increase this
            to avoid amplifying extremely small gradients. Defaults to 0.
        mode (str, optional): what to normalize.

            - "global": normalize the entire gradient, as if it was a single vector.

            - "param": normalize each param's gradient (default).

            - "channel": normalize gradient of each channel of each param.
        min_numel (int, optional):
            skips parameters with less than this many elements. This avoids the issue where
            parameters that have a single element always get set to the value of 1.
            Ignored when mode is 'global'.

    Example:
        >>> normalize_grad_(model.parameters())
    """
    _normalize_grad(
        (p.grad for p in params if p.grad is not None),
        norm_value = norm_value,
        ord = ord,
        min = min,
        mode = mode,
        min_numel = min_numel,
    )

class Normalize(OptimizerModule):
    def __init__(
        self,
        norm_value: float = 1,
        ord: float = 2,
        min: float = 0,
        mode: typing.Literal["global", "param", "channel"] = "global",
        min_numel=2,
    ):
        """Normalizes update to the given norm value.

        Args:
            norm_value (float, optional): value to normalize to. Defaults to 1.
            ord (float, optional): order of the norm. Defaults to 2.
            min (float, optional):
                won't normalize when gradient is below this norm, you can increase this
                to avoid amplifying extremely small gradients. Defaults to 0.
            mode (str, optional): what to normalize.

                - "global": normalize the entire gradient, as if it was a single vector.

                - "param": normalize each param's gradient (default).

                - "channel": normalize gradient of each channel of each param.
            min_numel (int, optional):
                skips parameters with less than this many elements. This avoids the issue where
                parameters that have a single element always get set to the value of 1.
                Ignored when mode is 'global'.
        """
        super().__init__({})
        self.norm_value = norm_value
        self.ord = ord
        self.min = min
        self.mode: typing.Literal["global", "param", "channel"] = mode
        self.min_numel = min_numel

    @torch.no_grad
    def _update(self, state, ascent):
        _normalize_grad(
            ascent,
            norm_value = self.norm_value,
            ord = self.ord,
            min = self.min,
            mode = self.mode,
            min_numel = self.min_numel,
        )


def _centralize_grad_(
    grads: abc.Iterable[torch.Tensor],
    mode: typing.Literal["global", "param", "channel"] = "channel",
    min_ndim=2,
    min_numel=2,
):
    if mode in ('param', 'channel'):
        if mode == 'channel': min_ndim = max(min_ndim, 2)
        for grad in grads:
            if grad.numel() >= min_numel and grad.ndim > min_ndim:
                if mode == 'channel':
                    grad -= grad.mean(dim=tuple(range(1, grad.ndim)), keepdim=True)
                else: # mode = 'param'
                    grad -= grad.mean()
    else:
        if not isinstance(grads, TensorList): grads = TensorList(grads)
        grads -= grads.mean()

@torch.no_grad
def centralize_grad_(
    params: abc.Iterable[torch.Tensor],
    mode: typing.Literal["global", "param", "channel"] = "channel",
    min_ndim=2,
    min_numel=2,
):
    """Centralizes gradients of an iterable of parameters.

    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to centralize.
        mode (str, optional): what to centralize.

            - "global": centralize the entire gradient (uses mean of entire gradient).

            - "param": centralize each param's gradient.

            - "channel": centralize gradient of each channel of each param (default).
        min_numel (int, optional):
            skips parameters with less than this many elements. This avoids negating updates for
            parameters that have a single element since subtracting mean always makes it 0.
            Ignored when mode is 'global'.
        min_ndim (int, optional):
            skips parameters with less than this many dimensions.
            bias usually has 1 dimension and you don't want to centralize it.
            Ignored when mode is 'global'.

    Reference:
        Yong, H., Huang, J., Hua, X., & Zhang, L. (2020). Gradient centralization: A new optimization technique for deep neural networks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16 (pp. 635-652). Springer International Publishing.

    Example:
        >>> centralize_grad_(model.parameters())
    """
    _centralize_grad_(
        (p.grad for p in params if p.grad is not None),
        mode = mode,
        min_ndim = min_ndim,
        min_numel = min_numel,
    )

class Centralize(OptimizerModule):
    def __init__(
        self,
        mode: typing.Literal["global", "param", "channel"] = "channel",
        min_ndim=2,
        min_numel=2,
    ):
        """Centralizes the update.

        Args:
            mode (str, optional): what to centralize.

                - "global": centralize the entire gradient (uses mean of entire gradient).

                - "param": centralize each param's gradient.

                - "channel": centralize gradient of each channel of each param (default).
            min_numel (int, optional):
                skips parameters with less than this many elements. This avoids negating updates for
                parameters that have a single element since subtracting mean always makes it 0.
                Ignored when mode is 'global'.
            min_ndim (int, optional):
                skips parameters with less than this many dimensions.
                bias usually has 1 dimension and you don't want to centralize it.
                Ignored when mode is 'global'.

        Reference:
            Yong, H., Huang, J., Hua, X., & Zhang, L. (2020). Gradient centralization: A new optimization technique for deep neural networks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16 (pp. 635-652). Springer International Publishing.
        """
        super().__init__({})
        self.mode: typing.Literal["global", "param", "channel"] = mode
        self.min_ndim = min_ndim
        self.min_numel = min_numel

    @torch.no_grad
    def _update(self, state, ascent):
        _centralize_grad_(
            ascent,
            mode = self.mode,
            min_ndim = self.min_ndim,
            min_numel = self.min_numel,
        )
