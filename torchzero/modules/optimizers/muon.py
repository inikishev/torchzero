from operator import itemgetter
import math
import warnings
from collections.abc import Iterable, Sequence
from typing import Literal

import torch

from ...core import Modular, TensorwiseTransform, Target, Transform
from ...utils import enable_compilation


def reverse_dims(t:torch.Tensor):
    return t.permute(*reversed(range(t.ndim)))

def _is_at_least_2d(p: torch.Tensor):
    if (p.ndim >= 2) and (p.size(0) > 1) and (p.size(1) > 1): return True
    return False

# stolen from:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
# actually at this stage its a frankenstein
@enable_compilation
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Applies to last 2 dims - so usually reverse_dims should be applied to G before and after.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# stolen from https://github.com/MarkTuddenham/Orthogonal-Optimisers.
# Tuddenham, M., Prügel-Bennett, A., & Hare, J. (2022).
# Orthogonalising gradients to speed up neural network optimisation. arXiv preprint arXiv:2202.07052.
@torch.no_grad
def _svd_orthogonalize(G: torch.Tensor, warn_fail=True) -> torch.Tensor:
    """
    Applies to first 2 dims and isn't batched - rest of dimensions are flattened.
    """
    X = G.view(G.shape[0], -1)

    t = False
    if X.size(0) > X.size(1):
        X = X.T
        t = True

    orth_X: torch.Tensor | None = None
    try:
        u, s, vt = torch.linalg.svd(X, full_matrices=False) # pylint:disable=not-callable
        orth_X = u @ vt
    except RuntimeError:
        # if warn: logging.warning('Failed to perform SVD, adding some noise.')
        try:
            u, s, v = torch.svd_lowrank(
                X,
                q=1,    # assume rank is at least 1
                M=1e-4 * X.mean() * torch.randn_like(X))
            orth_X = u @ v.T
        except RuntimeError:
            if warn_fail: warnings.warn(('Failed to perform SVD with noise,'
                            ' skipping gradient orthogonalisation'))
    if orth_X is not None:
        if t: orth_X = orth_X.T
        return orth_X.view_as(G)

    return G # fail


@torch.no_grad
def _dual_norm_correction(X: torch.Tensor, g: torch.Tensor, batch_first):
    """batch first means it applies to last 2 dims, otherwise to 1st two dims"""
    # this is from https://github.com/leloykun/adaptive-muon
    # Adaptive scaling,`(G * X).sum() * X` == (G.T @ X).trace() * X
    if batch_first: X = torch.einsum('...ij,...ij,...ab->...ab', g.type_as(X), X, X)
    else: X = torch.einsum('ij...,ij...,ab...->ab...', g.type_as(X), X, X)
    return X


# code from
# https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
def adjust_lr_for_muon(lr, param_shape):
    A, B = param_shape[:2]
    # We adjust the learning rate and weight decay based on the size of the parameter matrix
    # as describted in the paper
    adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr

def _orthogonalize_tensor(
    tensor: torch.Tensor,
    steps: int = 5,
    method: Literal["newton-schulz", "svd"] = "newton-schulz",
):
    if method == 'newton-schulz': return reverse_dims(zeropower_via_newtonschulz5(reverse_dims(tensor), steps)).type_as(tensor)
    if method == 'svd': return _svd_orthogonalize(tensor, False)
    raise ValueError(method)


def orthogonalize_grads_(
    params: Iterable[torch.Tensor],
    steps: int = 5,
    dual_norm_correction=False,
    method: Literal["newton-schulz", "svd"] = "newton-schulz",
):
    """Uses newton-Schulz iteration to compute the zeroth power / orthogonalization of gradients of an iterable of parameters.

    This sets gradients in-place. Applies along first 2 dims (expected to be `out_channels, in_channels`).

    Note that the Muon page says that embeddings and classifier heads should not be orthogonalized.
    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to orthogonalize.
        steps (int, optional):
            The number of Newton-Schulz iterations to run. Defaults to 5.
        dual_norm_correction (bool, optional):
            enables dual norm correction from https://github.com/leloykun/adaptive-muon. Defaults to False.
        method (str, optional):
            Newton-Schulz is very fast, SVD is extremely slow but can be slighly more precise.
    """
    for p in params:
        if (p.grad is not None) and _is_at_least_2d(p.grad):
            X = _orthogonalize_tensor(p.grad, steps, method)
            if dual_norm_correction: X = _dual_norm_correction(X, p.grad, batch_first=False)
            p.grad.set_(X.view_as(p)) # pyright:ignore[reportArgumentType]



class Orthogonalize(TensorwiseTransform):
    """Uses Newton-Schulz iteration or SVD to compute the zeroth power / orthogonalization of update along first 2 dims.

    To disable orthogonalization for a parameter, put it into a parameter group with "orthogonalize" = False.
    The Muon page says that embeddings and classifier heads should not be orthogonalized.
    Usually only matrix parameters that are directly used in matmuls should be orthogonalized.

    To make Muon, use Split with Adam on 1d params

    Args:
        ns_steps (int, optional):
            The number of Newton-Schulz iterations to run. Defaults to 5.
        adjust_lr (bool, optional):
            Enables LR adjustment based on parameter size from "Muon is Scalable for LLM Training". Defaults to False.
        dual_norm_correction (bool, optional):
            enables dual norm correction from https://github.com/leloykun/adaptive-muon. Defaults to False.
        method (str, optional):
            Newton-Schulz is very fast, SVD is extremely slow but can be slighly more precise.
        target (str, optional):
            what to set on var.


    Examples:
        standard Muon with Adam fallback

        .. code-block:: python

            opt = tz.Modular(
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

    Reference:
        Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, Jeremy Bernstein - Muon: An optimizer for hidden layers in neural networks (2024) https://github.com/KellerJordan/Muon
    """
    def __init__(self, ns_steps=5, adjust_lr=False, dual_norm_correction=False,
                 method: Literal['newton-schulz', 'svd'] = 'newton-schulz', target:Target='update'):
        defaults = dict(orthogonalize=True, ns_steps=ns_steps, dual_norm_correction=dual_norm_correction, adjust_lr=adjust_lr, method=method.lower())
        super().__init__(uses_grad=False, defaults=defaults, target=target)

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, loss, state, setting):
        orthogonalize, ns_steps, dual_norm_correction, adjust_lr, method = itemgetter(
            'orthogonalize', 'ns_steps', 'dual_norm_correction', 'adjust_lr', 'method')(setting)

        if not orthogonalize: return tensor

        if _is_at_least_2d(tensor):

            X = _orthogonalize_tensor(tensor, ns_steps, method)

            if dual_norm_correction:
                X = _dual_norm_correction(X, tensor, batch_first=False)

            if adjust_lr:
                X.mul_(adjust_lr_for_muon(1, param.shape))

            return X.view_as(param)

        return tensor


class DualNormCorrection(TensorwiseTransform):
    """Dual norm correction for dualizer based optimizers (https://github.com/leloykun/adaptive-muon).
    Orthogonalize already has this built in with the `dual_norm_correction` setting."""
    def __init__(self, target: Target='update'):
        super().__init__({}, uses_grad=True, target=target)

    def apply_tensor(self, tensor, param, grad, loss, state, setting):
        assert grad is not None
        if (tensor.ndim >= 2) and (tensor.size(0) > 1) and (tensor.size(1) > 1):
            return _dual_norm_correction(tensor, grad, batch_first=False)
        return tensor


class MuonAdjustLR(Transform):
    """LR adjustment for Muon from "Muon is Scalable for LLM Training" (https://github.com/MoonshotAI/Moonlight/tree/master).
    Orthogonalize already has this built in with the `adjust_lr` setting, however you might want to move this to be later in the chain."""
    def __init__(self, alpha: float = 1, target: Target='update'):
        defaults = dict(alpha=alpha)
        super().__init__(defaults=defaults, uses_grad=False, target=target)

    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        alphas = [s['alpha'] for s in settings]
        tensors_alphas = [(t, adjust_lr_for_muon(a, t.shape)) for t, a in zip(tensors, alphas) if _is_at_least_2d(t)]
        tensors = [i[0] for i in tensors_alphas]
        a = [i[1] for i in alphas]
        torch._foreach_mul_(tensors, a)
        return tensors
