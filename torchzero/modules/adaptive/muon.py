from operator import itemgetter
import math
from collections.abc import Iterable

import torch

from ...core import TensorTransform,  Transform
from ...linalg.orthogonalize import orthogonalize as _orthogonalize, OrthogonalizeMethod

def reverse_dims(t:torch.Tensor):
    return t.permute(*reversed(range(t.ndim)))

def _is_at_least_2d(p: torch.Tensor, channel_first:bool):
    if p.ndim < 2: return False
    if channel_first and (p.size(0) > 1) and (p.size(1) > 1): return True
    if (not channel_first) and (p.size(-2) > 1) and (p.size(-1) > 1): return True
    return False

def _orthogonalize_format(
    tensor: torch.Tensor,
    method: OrthogonalizeMethod,
    channel_first: bool,
):
    """orthogonalize either 1st two dims if channel first or last two otherwise"""
    if channel_first:
        return reverse_dims(_orthogonalize(reverse_dims(tensor), method=method))

    return _orthogonalize(tensor, method=method)

@torch.no_grad
def _dual_norm_correction(X: torch.Tensor, g: torch.Tensor, channel_first: bool):
    """``channel_first`` means it applies to first two dims, otherwise to last two dims"""
    # this is from https://github.com/leloykun/adaptive-muon
    # Adaptive scaling,`(G * X).sum() * X` == (G.T @ X).trace() * X
    if channel_first: X = torch.einsum('ij...,ij...,ab...->ab...', g.type_as(X), X, X)
    else: X = torch.einsum('...ij,...ij,...ab->...ab', g.type_as(X), X, X)
    return X


# code from
# https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
def adjust_lr_for_muon(lr, param_shape, channel_first:bool):
    if channel_first: A, B = param_shape[:2]
    else: A, B = param_shape[-2:]

    # We adjust the learning rate and weight decay based on the size of the parameter matrix
    # as describted in the paper
    adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


def orthogonalize_grads_(
    params: Iterable[torch.Tensor],
    dual_norm_correction=False,
    method: OrthogonalizeMethod = "newtonschulz",
    channel_first:bool=True,
):
    """Computes the zeroth power / orthogonalization of gradients of an iterable of parameters.

    This sets gradients in-place. Applies along first 2 dims (expected to be `out_channels, in_channels`).

    Note that the Muon page says that embeddings and classifier heads should not be orthogonalized.
    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to orthogonalize.
        dual_norm_correction (bool, optional):
            enables dual norm correction from https://github.com/leloykun/adaptive-muon. Defaults to False.
        method (str, optional):
            Newton-Schulz is very fast, SVD is extremely slow but can be slighly more precise.
        channel_first (bool, optional):
            if True, orthogonalizes along 1st two dimensions, otherwise along last 2. Other dimensions
            are considered batch dimensions.
    """
    for p in params:
        if (p.grad is not None) and _is_at_least_2d(p.grad, channel_first=channel_first):
            X = _orthogonalize_format(p.grad, method=method, channel_first=channel_first)
            if dual_norm_correction: X = _dual_norm_correction(X, p.grad, channel_first=False)
            p.grad.set_(X.view_as(p)) # pyright:ignore[reportArgumentType]



class Orthogonalize(TensorTransform):
    """Uses Newton-Schulz iteration or SVD to compute the zeroth power / orthogonalization of update along first 2 dims.

    To disable orthogonalization for a parameter, put it into a parameter group with "orthogonalize" = False.
    The Muon page says that embeddings and classifier heads should not be orthogonalized.
    Usually only matrix parameters that are directly used in matmuls should be orthogonalized.

    To make Muon, use Split with Adam on 1d params

    Args:
        adjust_lr (bool, optional):
            Enables LR adjustment based on parameter size from "Muon is Scalable for LLM Training". Defaults to False.
        dual_norm_correction (bool, optional):
            enables dual norm correction from https://github.com/leloykun/adaptive-muon. Defaults to False.
        method (str, optional):
            Newton-Schulz is very fast, SVD is slow but can be more precise.
        channel_first (bool, optional):
            if True, orthogonalizes along 1st two dimensions, otherwise along last 2. Other dimensions
            are considered batch dimensions.

    ## Examples:

    standard Muon with Adam fallback
    ```py
    opt = tz.Optimizer(
        model.head.parameters(),
        tz.m.Split(
            # apply muon only to 2D+ parameters
            filter = lambda t: t.ndim >= 2,
            true = [
                tz.m.HeavyBall(),
                tz.m.Orthogonalize(),
                tz.m.LR(1e-2),
            ],
            false = tz.m.Adam()
        ),
        tz.m.LR(1e-2)
    )
    ```

    Reference:
        Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, Jeremy Bernstein - Muon: An optimizer for hidden layers in neural networks (2024) https://github.com/KellerJordan/Muon
    """
    def __init__(self, adjust_lr=False, dual_norm_correction=False,
                 method: OrthogonalizeMethod = 'newtonschulz', channel_first:bool=True):
        defaults = dict(orthogonalize=True, dual_norm_correction=dual_norm_correction, adjust_lr=adjust_lr, method=method.lower(), channel_first=channel_first)
        super().__init__(defaults=defaults)

    @torch.no_grad
    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        orthogonalize, dual_norm_correction, adjust_lr, method, channel_first = itemgetter(
            'orthogonalize', 'dual_norm_correction', 'adjust_lr', 'method', 'channel_first')(setting)

        if not orthogonalize: return tensor

        if _is_at_least_2d(tensor, channel_first=channel_first):

            X = _orthogonalize_format(tensor, method, channel_first=channel_first)

            if dual_norm_correction:
                X = _dual_norm_correction(X, tensor, channel_first=channel_first)

            if adjust_lr:
                X.mul_(adjust_lr_for_muon(1, param.shape, channel_first=channel_first))

            return X.view_as(param)

        return tensor


class DualNormCorrection(TensorTransform):
    """Dual norm correction for dualizer based optimizers (https://github.com/leloykun/adaptive-muon).
    Orthogonalize already has this built in with the `dual_norm_correction` setting."""
    def __init__(self, channel_first: bool = True):
        defaults = dict(channel_first=channel_first)
        super().__init__(defaults)

    @torch.no_grad
    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        assert grad is not None
        if (tensor.ndim >= 2) and (tensor.size(0) > 1) and (tensor.size(1) > 1):
            return _dual_norm_correction(tensor, grad, channel_first=setting["channel_first"])
        return tensor


class MuonAdjustLR(Transform):
    """LR adjustment for Muon from "Muon is Scalable for LLM Training" (https://github.com/MoonshotAI/Moonlight/tree/master).
    Orthogonalize already has this built in with the ``adjust_lr`` setting, however you might want to move this to be later in the chain."""
    def __init__(self, channel_first: bool = True, alpha: float = 1):
        defaults = dict(channel_first=channel_first, alpha=alpha)
        super().__init__(defaults=defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        alphas = [s['alpha'] for s in settings]
        channel_first = [s["channel_first=channel_first"] for s in settings]
        tensors_alphas = [
            (t, adjust_lr_for_muon(a, t.shape, cf)) for t, a, cf in zip(tensors, alphas, channel_first) if _is_at_least_2d(t, channel_first=cf)
        ]
        tensors = [i[0] for i in tensors_alphas]
        a = [i[1] for i in alphas]
        torch._foreach_mul_(tensors, a)
        return tensors
