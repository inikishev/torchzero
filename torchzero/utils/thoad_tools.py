import itertools
from collections.abc import Callable
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import torch

from .python_tools import LazyLoader

lazy_thoad = LazyLoader("thoad")
if TYPE_CHECKING:
    import thoad
    lazy_thoad = cast(thoad, lazy_thoad)

def thoad_single_tensor(
    ctrl: "thoad.Controller",
    params: list[torch.Tensor],
    order: int
) -> torch.Tensor:
    """treats params as if they were concatenated into a vector."""

    if not all(p.requires_grad for p in params):
        raise ValueError("All parameters must have requires_grad=True")

    if order < 1:
        raise ValueError("Order must be at least 1")

    # we need parameter sizes and total size N
    # final tensor is (N, N, ..., N) with `order` dimensions.
    param_numels = [p.numel() for p in params]
    total_params = sum(param_numels)

    final_shape = (total_params,) * order
    p = params[0]
    T = torch.zeros(final_shape, device=p.device, dtype=p.dtype)

    # start/end indices for each parameter in the flattened vector.
    offsets = torch.cumsum(torch.tensor([0] + param_numels), dim=0)

    # for order=2 this iterates through (p0,p0), (p0,p1), (p1,p0), (p1,p1), etc.
    param_indices = range(len(params))
    for block_indices in itertools.product(param_indices, repeat=order):

        block_params = tuple(params[i] for i in block_indices)
        block_tensor, _ = ctrl.fetch_hgrad(variables=block_params) # (1, *p1.shape, *p2.shape, ...).
        block_tensor = block_tensor.squeeze(0) # (*p1.shape, *p2.shape, ...)

        # convert (*p1.shape, *p2.shape) to (p1.numel(), p2.numel())
        block_flat_shape = tuple(param_numels[i] for i in block_indices)
        block_tensor_flat = block_tensor.reshape(block_flat_shape)

        # place the flattened block into T
        slicing = tuple(
            slice(offsets[i], offsets[i+1]) for i in block_indices
        )
        T[slicing] = block_tensor_flat

    ctrl.clear()
    return T

def thoad_derivatives(
    ctrl: "thoad.Controller",
    params: list[torch.Tensor],
    order: int,
):
    """returns all derivatives up to ``order`` in ascending order, all as single tensors
    as if parameters were concatenated to a vector"""
    return [thoad_single_tensor(ctrl, params, o) for o in range(1, order+1)]