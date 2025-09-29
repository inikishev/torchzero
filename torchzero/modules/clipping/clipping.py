import math
from collections.abc import Iterable, Sequence
from operator import itemgetter
from typing import Literal

import torch

from ...core import Module,  TensorTransform
from ...utils import Metrics, NumberList, TensorList
from ...utils.metrics import _METRICS


def clip_grad_value_(params: Iterable[torch.Tensor], value: float):
    """Clips gradient of an iterable of parameters at specified value.
    Gradients are modified in-place.
    Args:
        params (Iterable[Tensor]): iterable of tensors with gradients to clip.
        value (float or int): maximum allowed value of gradient
    """
    grads = [p.grad for p in params if p.grad is not None]
    torch._foreach_clamp_min_(grads, -value)
    torch._foreach_clamp_max_(grads, value)

def _clip_norm_(
    tensors_: TensorList,
    min: float | NumberList | None,
    max: float | NumberList | None,
    norm_value: float | NumberList | None,
    ord: Metrics,
    dim: int | Sequence[int] | Literal["global"] | None,
    inverse_dims: bool,
    min_size: int,
) -> TensorList:
    """generic function that can clip norm or normalize"""
    if norm_value is not None:
        if min is not None or max is not None:
            raise ValueError(f'if norm_value is given then min and max must be None got {min = }; {max = }')

        # if dim is None: return tensors_.mul_(norm_value / tensors_.norm(ord=ord))
        if dim == 'global': return tensors_.mul_(norm_value / tensors_.global_metric(ord))

    # if dim is None: return tensors_.clip_norm_(min,max,tensorwise=True,ord=ord)
    if dim == 'global': return tensors_.clip_norm_(min,max,tensorwise=False,ord=ord)

    muls = []
    tensors_to_mul = []
    if isinstance(dim, int): dim = (dim, )

    for i, tensor in enumerate(tensors_):
        # remove dimensions that overflow tensor.ndim or are too small
        if tensor.ndim == 0: tensor = tensor.unsqueeze(0)
        if dim is None: dim = list(range(tensor.ndim))
        real_dim = [d for d in dim if d < tensor.ndim]
        if inverse_dims: real_dim = [d for d in range(tensor.ndim) if d not in real_dim]
        if len(real_dim) == 0: continue
        size = math.prod(tensor.size(d) for d in real_dim)
        if size < min_size: continue

        if isinstance(ord, str):
            norm = _METRICS[ord].evaluate_tensor(tensor, dim=real_dim, keepdim=True)
        else:
            norm: torch.Tensor = torch.linalg.vector_norm(tensor, ord=ord, dim=real_dim, keepdim=True) # pylint:disable=not-callable

        if norm.numel() == 1 and norm == 0: continue
        norm = torch.where(norm <= 1e-12, 1, norm)

        # normalize = True, perform normalization
        norm_v = norm_value[i] if isinstance(norm_value, (list,tuple)) else norm_value
        if norm_v is not None:
            mul = norm_v / norm

        # else clip to min and max norms
        else:
            minv = min[i] if isinstance(min, (list,tuple)) else min
            maxv = max[i] if isinstance(max, (list,tuple)) else max

            mul = 1
            if minv is not None:
                mul_to_min = (minv / norm).clamp(min=1)
                mul *= mul_to_min

            if maxv is not None:
                mul_to_max = (maxv / norm).clamp(max=1)
                mul *= mul_to_max

        muls.append(mul)
        tensors_to_mul.append(tensor)

    if len(muls) > 0:


        torch._foreach_mul_(tensors_to_mul, muls)
    return tensors_


def clip_grad_norm_(
    params: Iterable[torch.Tensor],
    max_norm: float | None,
    ord: Metrics = 2,
    dim: int | Sequence[int] | Literal["global"] | None = None,
    inverse_dims: bool = False,
    min_size: int = 2,
    min_norm: float | None = None,
):
    """Clips gradient of an iterable of parameters to specified norm value.
    Gradients are modified in-place.

    Args:
        params (Iterable[torch.Tensor]): parameters with gradients to clip.
        max_norm (float): value to clip norm to.
        ord (float, optional): norm order. Defaults to 2.
        dim (int | Sequence[int] | str | None, optional):
            calculates norm along those dimensions.
            If list/tuple, tensors are normalized along all dimensios in `dim` that they have.
            Can be set to "global" to normalize by global norm of all gradients concatenated to a vector.
            Defaults to None.
        min_size (int, optional):
            minimal size of a dimension to normalize along it. Defaults to 1.
    """
    grads = TensorList(p.grad for p in params if p.grad is not None)
    _clip_norm_(grads, min=min_norm, max=max_norm, norm_value=None, ord=ord, dim=dim, inverse_dims=inverse_dims, min_size=min_size)


def normalize_grads_(
    params: Iterable[torch.Tensor],
    norm_value: float,
    ord: Metrics = 2,
    dim: int | Sequence[int] | Literal["global"] | None = None,
    inverse_dims: bool = False,
    min_size: int = 1,
):
    """Normalizes gradient of an iterable of parameters to specified norm value.
    Gradients are modified in-place.

    Args:
        params (Iterable[torch.Tensor]): parameters with gradients to clip.
        norm_value (float): value to clip norm to.
        ord (float, optional): norm order. Defaults to 2.
        dim (int | Sequence[int] | str | None, optional):
            calculates norm along those dimensions.
            If list/tuple, tensors are normalized along all dimensios in `dim` that they have.
            Can be set to "global" to normalize by global norm of all gradients concatenated to a vector.
            Defaults to None.
        inverse_dims (bool, optional):
            if True, the `dims` argument is inverted, and all other dimensions are normalized.
        min_size (int, optional):
            minimal size of a dimension to normalize along it. Defaults to 1.
    """
    grads = TensorList(p.grad for p in params if p.grad is not None)
    _clip_norm_(grads, min=None, max=None, norm_value=norm_value, ord=ord, dim=dim, inverse_dims=inverse_dims, min_size=min_size)


class ClipValue(TensorTransform):
    """Clips update magnitude to be within ``(-value, value)`` range.

    Args:
        value (float): value to clip to.
        target (str): refer to ``target argument`` in documentation.

    Examples:

    Gradient clipping:
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.ClipValue(1),
        tz.m.Adam(),
        tz.m.LR(1e-2),
    )
    ```

    Update clipping:
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Adam(),
        tz.m.ClipValue(1),
        tz.m.LR(1e-2),
    )
    ```

    """
    def __init__(self, value: float):
        defaults = dict(value=value)
        super().__init__(defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        value = [s['value'] for s in settings]
        return TensorList(tensors).clip_([-v for v in value], value)

class ClipNorm(TensorTransform):
    """Clips update norm to be no larger than ``value``.

    Args:
        max_norm (float): value to clip norm to.
        ord (float, optional): norm order. Defaults to 2.
        dim (int | Sequence[int] | str | None, optional):
            calculates norm along those dimensions.
            If list/tuple, tensors are normalized along all dimensios in `dim` that they have.
            Can be set to "global" to normalize by global norm of all gradients concatenated to a vector.
            Defaults to None.
        inverse_dims (bool, optional):
            if True, the `dims` argument is inverted, and all other dimensions are normalized.
        min_size (int, optional):
            minimal numer of elements in a parameter or slice to clip norm. Defaults to 1.
        target (str, optional):
            what this affects.

    Examples:

    Gradient norm clipping:
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.ClipNorm(1),
        tz.m.Adam(),
        tz.m.LR(1e-2),
    )
    ```

    Update norm clipping:
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Adam(),
        tz.m.ClipNorm(1),
        tz.m.LR(1e-2),
    )
    ```
    """
    def __init__(
        self,
        max_norm: float,
        ord: Metrics = 2,
        dim: int | Sequence[int] | Literal["global"] | None = None,
        inverse_dims: bool = False,
        min_size: int = 1,
    ):
        defaults = dict(max_norm=max_norm,ord=ord,dim=dim,min_size=min_size,inverse_dims=inverse_dims)
        super().__init__(defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        max_norm = NumberList(s['max_norm'] for s in settings)
        ord, dim, min_size, inverse_dims = itemgetter('ord', 'dim', 'min_size', 'inverse_dims')(settings[0])
        _clip_norm_(
            tensors_ = TensorList(tensors),
            min = 0,
            max = max_norm,
            norm_value = None,
            ord = ord,
            dim = dim,
            inverse_dims=inverse_dims,
            min_size = min_size,
        )
        return tensors

class Normalize(TensorTransform):
    """Normalizes the update.

    Args:
        norm_value (float): desired norm value.
        ord (float, optional): norm order. Defaults to 2.
        dim (int | Sequence[int] | str | None, optional):
            calculates norm along those dimensions.
            If list/tuple, tensors are normalized along all dimensios in `dim` that they have.
            Can be set to "global" to normalize by global norm of all gradients concatenated to a vector.
            Defaults to None.
        inverse_dims (bool, optional):
            if True, the `dims` argument is inverted, and all other dimensions are normalized.
        min_size (int, optional):
            minimal size of a dimension to normalize along it. Defaults to 1.
        target (str, optional):
            what this affects.

    Examples:
    Gradient normalization:
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Normalize(1),
        tz.m.Adam(),
        tz.m.LR(1e-2),
    )
    ```

    Update normalization:

    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Adam(),
        tz.m.Normalize(1),
        tz.m.LR(1e-2),
    )
    ```
    """
    def __init__(
        self,
        norm_value: float = 1,
        ord: Metrics = 2,
        dim: int | Sequence[int] | Literal["global"] | None = None,
        inverse_dims: bool = False,
        min_size: int = 1,
    ):
        defaults = dict(norm_value=norm_value,ord=ord,dim=dim,min_size=min_size, inverse_dims=inverse_dims)
        super().__init__(defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        norm_value = NumberList(s['norm_value'] for s in settings)
        ord, dim, min_size, inverse_dims = itemgetter('ord', 'dim', 'min_size', 'inverse_dims')(settings[0])

        _clip_norm_(
            tensors_ = TensorList(tensors),
            min = None,
            max = None,
            norm_value = norm_value,
            ord = ord,
            dim = dim,
            inverse_dims=inverse_dims,
            min_size = min_size,
        )

        return tensors


def _centralize_(
    tensors_: TensorList,
    dim: int | Sequence[int] | Literal["global"] | None,
    min_size: int,
    inverse_dims: bool,
) -> TensorList:
    """generic function that can clip norm or normalize"""
    if dim == 'global': return tensors_.sub_(tensors_.global_mean().item())

    subs = []
    tensors_to_sub = []
    if isinstance(dim, int): dim = (dim, )

    for tensor in tensors_:
        # remove dimensions that overflow tensor.ndim or are too small
        if dim is None: dim = list(range(tensor.ndim))
        real_dim = [d for d in dim if d < tensor.ndim]
        if inverse_dims: real_dim = [d for d in range(tensor.ndim) if d not in real_dim]
        if len(real_dim) == 0: continue
        size = math.prod(tensor.size(d) for d in real_dim)
        if size < min_size: continue

        mean: torch.Tensor = torch.mean(tensor, dim=real_dim, keepdim=True)
        if mean.numel() == 1 and mean == 0: continue

        subs.append(mean)
        tensors_to_sub.append(tensor)

    if len(subs) > 0:
        torch._foreach_sub_(tensors_to_sub, subs)

    return tensors_


class Centralize(TensorTransform):
    """Centralizes the update.

    Args:
        dim (int | Sequence[int] | str | None, optional):
            calculates norm along those dimensions.
            If list/tuple, tensors are centralized along all dimensios in `dim` that they have.
            Can be set to "global" to centralize by global mean of all gradients concatenated to a vector.
            Defaults to None.
        inverse_dims (bool, optional):
            if True, the `dims` argument is inverted, and all other dimensions are centralized.
        min_size (int, optional):
            minimal size of a dimension to normalize along it. Defaults to 1.

    Examples:

    Standard gradient centralization:
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Centralize(dim=0),
        tz.m.LR(1e-2),
    )
    ```

    References:
    - Yong, H., Huang, J., Hua, X., & Zhang, L. (2020). Gradient centralization: A new optimization technique for deep neural networks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16 (pp. 635-652). Springer International Publishing. https://arxiv.org/abs/2004.01461
    """
    def __init__(
        self,
        dim: int | Sequence[int] | Literal["global"] | None = None,
        inverse_dims: bool = False,
        min_size: int = 2,
    ):
        defaults = dict(dim=dim,min_size=min_size,inverse_dims=inverse_dims)
        super().__init__(defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        dim, min_size, inverse_dims = itemgetter('dim', 'min_size', 'inverse_dims')(settings[0])

        _centralize_(tensors_ = TensorList(tensors), dim=dim, inverse_dims=inverse_dims, min_size=min_size)

        return tensors

