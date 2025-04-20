from operator import itemgetter
from typing import Literal
from collections.abc import Iterable, Sequence

import torch

from ...core import Module, Target, Transform
from ...utils import NumberList, TensorList, generic_eq


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
    ord: float,
    dim: int | Sequence[int] | Literal["global"] | None,
    min_size: int,
) -> TensorList:
    """generic function that can clip norm or normalize"""
    if norm_value is not None:
        if min is not None or max is not None:
            raise ValueError(f'if norm_value is given then min and max must be None got {min = }; {max = }')

        if dim is None: return tensors_.mul_(norm_value / tensors_.norm(ord=ord))
        if dim == 'global': return tensors_.mul_(norm_value / tensors_.global_vector_norm(ord=ord))

    if dim is None: return tensors_.clip_norm_(min,max,tensorwise=True,ord=ord)
    if dim == 'global': return tensors_.clip_norm_(min,max,tensorwise=False,ord=ord)

    muls = []
    tensors_to_mul = []
    if isinstance(dim, int): dim = (dim, )

    for tensor in tensors_:
        # remove dimensions that overflow tensor.ndim or are too small
        real_dim = [d for d in dim if (d < tensor.ndim) and (tensor.shape[d] >= min_size)]
        if len(real_dim) == 0: continue

        norm: torch.Tensor = torch.linalg.vector_norm(tensor, ord=ord, dim=dim, keepdim=True) # pylint:disable=not-callable
        if norm == 0: continue

        # normalize = True, perform normalization
        if norm_value is not None:
            mul = norm_value / norm

        # else clip to min and max norms
        else:
            mul = 0
            if min is not None:
                mul_to_min = (min / norm).clamp_(min=1) # type:ignore
                mul *= mul_to_min

            if max is not None:
                mul_to_max = (max / norm).clamp_(max=1) # type:ignore
                mul *= mul_to_max

        muls.append(mul)
        tensors_to_mul.append(tensor)

    if len(muls) > 0: torch._foreach_mul_(tensors_to_mul, muls)
    return tensors_


def clip_grad_norm_(
    params: Iterable[torch.Tensor],
    max_norm: float | None,
    ord: float = 2,
    dim: int | Sequence[int] | Literal["global"] | None = None,
    min_size: int = 1,
    min_norm: float | None = None,
):
    """Clips gradient of an iterable of parameters to specified norm value.
    Gradients are modified in-place.

    Args:
        params (Iterable[torch.Tensor]): parameters with gradients to clip.
        value (float): value to clip norm to.
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
    _clip_norm_(grads, min=min_norm, max=max_norm, norm_value=None, ord=ord, dim=dim, min_size=min_size)


def normalize_grads_(
    params: Iterable[torch.Tensor],
    norm_value: float,
    ord: float = 2,
    dim: int | Sequence[int] | Literal["global"] | None = None,
    min_size: int = 1,
):
    """Normalizes gradient of an iterable of parameters to specified norm value.
    Gradients are modified in-place.

    Args:
        params (Iterable[torch.Tensor]): parameters with gradients to clip.
        value (float): value to clip norm to.
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
    _clip_norm_(grads, min=None, max=None, norm_value=norm_value, ord=ord, dim=dim, min_size=min_size)


class ClipValue(Transform):
    def __init__(self, value: float, target: Target = 'update'):
        defaults = dict(value=value)
        super().__init__(defaults, target)

    @torch.no_grad
    def transform(self, target, vars):
        value = self.get_settings('value', params=vars.params)
        return TensorList(target).clip_([-v for v in value], value)

class ClipNorm(Transform):
    """Clips update norm to a value.

    Args:
        value (float): value to clip norm to.
        ord (float, optional): norm order. Defaults to 2.
        dim (int | Sequence[int] | str | None, optional):
            calculates norm along those dimensions.
            If list/tuple, tensors are normalized along all dimensios in `dim` that they have.
            Can be set to "global" to normalize by global norm of all gradients concatenated to a vector.
            Defaults to None.
        min_size (int, optional):
            minimal size of a dimension to normalize along it. Defaults to 1.
    """
    def __init__(
        self,
        max_norm: float,
        ord: float = 2,
        dim: int | Sequence[int] | Literal["global"] | None = None,
        min_size: int = 1,
        target: Target = "update",
        min_norm: float | None = None,
    ):
        defaults = dict(max_norm=max_norm,ord=ord,dim=dim,min_size=min_size,min_norm=min_norm)
        super().__init__(defaults, target)

    @torch.no_grad
    def transform(self, target, vars):
        max_norm, min_norm = self.get_settings('max_norm', 'min_norm', params=vars.params, cls=NumberList)
        ord, dim, min_size = itemgetter('ord', 'dim', 'min_size')(self.defaults)
        _clip_norm_(
            tensors_ = TensorList(target),
            min = min_norm if min_norm[0] is not None else None,
            max = max_norm if max_norm[0] is not None else None,
            norm_value = None,
            ord = ord,
            dim = dim,
            min_size = min_size,
        )
        return target

class Normalize(Transform):
    """Normalizes the update.

    Args:
        value (float): desired norm value.
        ord (float, optional): norm order. Defaults to 2.
        dim (int | Sequence[int] | str | None, optional):
            calculates norm along those dimensions.
            If list/tuple, tensors are normalized along all dimensios in `dim` that they have.
            Can be set to "global" to normalize by global norm of all gradients concatenated to a vector.
            Defaults to None.
        min_size (int, optional):
            minimal size of a dimension to normalize along it. Defaults to 1.
    """
    def __init__(
        self,
        norm_value: float = 1,
        ord: float = 2,
        dim: int | Sequence[int] | Literal["global"] | None = None,
        min_size: int = 1,
        target: Target = "update",
    ):
        defaults = dict(norm_value=norm_value,ord=ord,dim=dim,min_size=min_size)
        super().__init__(defaults, target)

    @torch.no_grad
    def transform(self, target, vars):
        norm_value = self.get_settings('norm_value', params=vars.params, cls=NumberList)
        ord, dim, min_size = itemgetter('ord', 'dim', 'min_size')(self.defaults)

        _clip_norm_(
            tensors_ = TensorList(target),
            min = None,
            max = None,
            norm_value = norm_value,
            ord = ord,
            dim = dim,
            min_size = min_size,
        )

        return target
