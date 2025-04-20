import math
import warnings
from collections.abc import Iterable, Sequence
from typing import Literal

import torch

from ...core import Modular, ParameterwiseTransform, Target
from ...utils import _maybe_compile


def reverse_dims(t:torch.Tensor):
    return t.permute(*reversed(range(t.ndim)))

# from ...utils.compile import maybe_compile

# stolen from:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@_maybe_compile
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


@torch.no_grad
def _svd_orthogonalize(G: torch.Tensor, warn_fail=True) -> torch.Tensor:
    """stolen from https://github.com/MarkTuddenham/Orthogonal-Optimisers.

    Applies to first 2 dims and isn't batched - rest of dimensions are flattened.

    Tuddenham, M., Prügel-Bennett, A., & Hare, J. (2022).
    Orthogonalising gradients to speed up neural network optimisation. arXiv preprint arXiv:2202.07052.
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
def _adaptive_scaling(X: torch.Tensor, g: torch.Tensor, batch_first):
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
    adaptive=True,
    method: Literal["newton-schulz", "svd"] = "newton-schulz",
):
    """Uses newton-Schulz iteration to compute the zeroth power / orthogonalization of gradients of an iterable of parameters.

    This sets gradients in-place. Applies to first 2 dims.

    Note that the Muon page says that embeddings and classifier heads should not be orthogonalized.
    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to orthogonalize.
        steps (int):
            The number of Newton-Schulz iterations to run. (5 is probably always enough). Defaults to 5.
        adaptive (bool, optional):
            Enables adaptation to scale of gradients (from https://github.com/leloykun/adaptive-muon). Defaults to False.
    """
    for p in params:
        if (p.grad is not None) and (p.grad.ndim >= 2) and (p.grad.size(0) > 1) and (p.grad.size(1) > 1):
            X = _orthogonalize_tensor(p.grad, steps, method)
            if adaptive: X = _adaptive_scaling(X, p.grad, batch_first=False)
            p.grad.set_(X.view_as(p)) # pyright:ignore[reportArgumentType]



class Orthogonalize(ParameterwiseTransform):
    """Uses Newton-Schulz iteration or SVD to compute the zeroth power / orthogonalization of gradients.
    Applies to first 2 dims.

    To disable orthogonalization for a parameter, put it into a parameter group with "orthogonalize" = False.
    The Muon page says that embeddings and classifier heads should not be orthogonalized.

    The orthogonalization code is taken from https://github.com/KellerJordan/Muon.

    Note that unlike this module, Muon also uses Adam for gradients that are not orthogonalized.
    You can achive the same by TODO....

    However not using Adam, or putting Adam module after this to apply it to ALL updates, both seem
    to work quite well too.

    Args:
        ns_steps (int, optional):
            The number of Newton-Schulz iterations to run. (5 is probably always enough). Defaults to 5.
        adaptive (bool, optional):
            Enables adaptation to scale of gradients (from https://github.com/leloykun/adaptive-muon). Defaults to True.
        target (str, optional):
            what to set on vars.
    """
    def __init__(self, ns_steps=5, adaptive=True, adjust_lr=True,
                 method: Literal['newton-schulz', 'svd'] = 'newton-schulz', target:Target='update'):
        defaults = dict(orthogonalize=True, ns_steps=ns_steps, adaptive=adaptive, adjust_lr=adjust_lr, method=method)
        super().__init__(requires_grad=False, defaults=defaults, target=target)

    @torch.no_grad
    def transform(self, target, param, grad, vars):
        settings = self.settings[param]
        if not settings['orthogonalize']: return target

        if (target.ndim >= 2) and (target.size(0) > 1) and (target.size(1) > 1):

            X = _orthogonalize_tensor(target, settings['ns_steps'], settings['method'])

            if settings['adaptive']:
                X = _adaptive_scaling(X, target, batch_first=False)

            if settings['adjust_lr']:
                X.mul_(adjust_lr_for_muon(1, param.shape))

            return X.view_as(param)

        return target


class MuonAdaptiveScaling(ParameterwiseTransform):
    """adaptive scaling for Muon"""
    def __init__(self, target: Target='update'):
        super().__init__(requires_grad=True, target=target)

    def transform(self, target, param, grad, vars):
        assert grad is not None
        if (target.ndim >= 2) and (target.size(0) > 1) and (target.size(1) > 1):
            return _adaptive_scaling(target, grad, batch_first=False)
        return target

class MuonAdjustLR(ParameterwiseTransform):
    """LR adjustment for Muon"""
    def __init__(self, target: Target='update'):
        super().__init__(requires_grad=False, target=target)

    def transform(self, target, param, grad, vars):
        if (target.ndim >= 2) and (target.size(0) > 1) and (target.size(1) > 1):
            return target.mul_(adjust_lr_for_muon(1, param.shape))
        return target



class DualNormCorrection(ParameterwiseTransform):
    """Dual norm correction from https://github.com/leloykun/adaptive-muon.

    Description from the page:

    Single-line modification to any (dualizer-based) optimizer that allows
    the optimizer to adapt to the scale of the gradients as they change during training.
    This is done by scaling the dualized gradient by the clipped dual norm of the original gradient.
    """
    def __init__(self, adaptive_scale_min: int | None = -1, adaptive_scale_max: int | None = 1):
        defaults = dict(adaptive_scale_min = adaptive_scale_min, adaptive_scale_max = adaptive_scale_max)
        super().__init__(requires_grad=True, defaults=defaults)

    @torch.no_grad
    def transform(self, target, param, grad, vars):
        assert grad is not None
        settings = self.settings[param]
        adaptive_scale_min, adaptive_scale_max = settings['adaptive_scale_min'], settings['adaptive_scale_max']

        if len([s for s in target.shape if s > 1]) >= 2:
            scale = torch.einsum('ij...,ij...->', grad, target)
            if adaptive_scale_min is not None or adaptive_scale_max is not None:
                scale = scale.clip(adaptive_scale_min, adaptive_scale_max)
            target *= scale

        return target


