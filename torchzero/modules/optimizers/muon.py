import warnings
from collections.abc import Iterable, Sequence
from typing import Literal
import torch

from ...core import Modular, ParameterwiseTransform, Target
from ...utils import _maybe_compile

# from ...utils.compile import maybe_compile

# stolen from:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@_maybe_compile
@torch.no_grad
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int, dims = (0, 1)) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Newton-Schulz iteration code stolen from https://github.com/KellerJordan/Muon

    Keller Jordan and Yuchen Jin and Vlado Boza and You Jiacheng and Franz Cecista and Laker Newhouse and Jeremy Bernstein.
    Muon: An optimizer for hidden layers in neural networks (2024). URL: https://kellerjordan.github.io/posts/muon
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    d0, d1 = dims

    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()

    if dims != (0, 1): X = X.swapdims(d0, 0).swapdims(d1, 1)
    t = False
    if X.size(0) > X.size(1):
        t = True
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(0,1), keepdim=True) + 1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if t: X = X.T
    if dims != (0, 1): X = X.swapdims(d0, 0).swapdims(d1, 1)
    return X

@torch.no_grad
def _svd_orthogonalize_(G: torch.Tensor, dims = (0, 1), warn_fail=True) -> torch.Tensor:
    """stolen from https://github.com/MarkTuddenham/Orthogonal-Optimisers

    Tuddenham, M., Prügel-Bennett, A., & Hare, J. (2022).
    Orthogonalising gradients to speed up neural network optimisation. arXiv preprint arXiv:2202.07052.
    """
    d0, d1 = dims
    X = G
    if dims != (0, 1): X = X.swapdims(d0, 0).swapdims(d1, 1)
    X_full = X
    X = X_full.view(X_full.shape[0], -1)

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
        orth_X = orth_X.view_as(X_full)
        if dims != (0, 1): orth_X = orth_X.swapdims(d0, 0).swapdims(d1, 1)
        return orth_X.view_as(G)

    return G # fail


@torch.no_grad
def _adaptive_scaling(X: torch.Tensor, g: torch.Tensor, dims = (0, 1)):
    d0, d1 = dims
    if dims != (0, 1): X = X.swapdims(d0, 0).swapdims(d1, 1)

    t = False
    # is this needed?
    if X.size(0) > X.size(1):
        X = X.swapdims(0,1)
        g = g.swapdims(0,1)
        t = True

    # this is from https://github.com/leloykun/adaptive-muon
    # Adaptive scaling,`(G * X).sum() * X` == (G.T @ X).trace() * X
    X = torch.einsum('ij...,ij...,ab...->ab...', g.type_as(X), X, X)
    if t: X = X.swapdims(0,1)
    if dims != (0, 1): X = X.swapdims(d0, 0).swapdims(d1, 1)

    return X

def orthogonalize_grads_(params: Iterable[torch.Tensor], steps: int = 5, adaptive=True, method: Literal['newton-schulz', 'svd'] = 'newton-schulz', dims = (0, 1)):
    """Uses newton-Schulz iteration to compute the zeroth power / orthogonalization of gradients of an iterable of parameters.

    This sets gradients in-place.

    Note that the Muon page says that embeddings and classifier heads should not be orthogonalized.
    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to orthogonalize.
        steps (int):
            The number of Newton-Schulz iterations to run. (5 is probably always enough). Defaults to 5.
        adaptive (bool, optional):
            Enables adaptation to scale of gradients (from https://github.com/leloykun/adaptive-muon). Defaults to False.
        compiled (bool, optional):
            Uses compiled newton-Schulz iteration function. Faster but won't work on windows. Defaults to True.


    """
    for p in params:
        if p.grad is not None and len([s for s in p.grad.shape if s > 1]) >= 2:

            if method == 'newton-schulz': X = zeropower_via_newtonschulz5(p.grad, steps, dims=dims)
            elif method == 'svd': X = _svd_orthogonalize_(p.grad, dims=dims, warn_fail=False)
            else: raise ValueError(method)
            if adaptive: X = _adaptive_scaling(X, p.grad, dims)
            p.grad.set_(X.view_as(p.grad)) # pyright:ignore[reportArgumentType]



class Orthogonalize(ParameterwiseTransform):
    """Uses Newton-Schulz iteration or SVD to compute the zeroth power / orthogonalization of gradients of an iterable of parameters.

    To disable orthogonalization for a parameter, put it into a parameter group with "orthogonalize" = False.
    The Muon page says that embeddings and classifier heads should not be orthogonalized.

    The orthogonalization code is taken from https://github.com/KellerJordan/Muon.

    Note that unlike this module, Muon also uses Adam for gradients that are not orthogonalized.
    You can achive the same by TODO....

    However not using Adam, or putting Adam module after this to apply it to ALL updates, both seem
    to work quite well too.

    Args:
        ns_steps (int, optional):
            The number of Newton-Schulz iterations to run. (56 is probably always enough). Defaults to 5.
        adaptive (bool, optional):
            Enables adaptation to scale of gradients (from https://github.com/leloykun/adaptive-muon). Defaults to True.
        target (str, optional):
            what to set on vars.
    """
    def __init__(self, ns_steps=5, adaptive=True, method: Literal['newton-schulz', 'svd'] = 'newton-schulz', dims = (0,1), target:Target='update'):
        defaults = dict(orthogonalize=True, ns_steps=ns_steps, adaptive=adaptive, method=method, dims=dims)
        super().__init__(requires_grad=adaptive, defaults=defaults, target=target)

    @torch.no_grad
    def transform(self, target, param, grad, vars):
        settings = self.settings[param]
        if not settings['orthogonalize']: return target

        if len([s for s in target.shape if s > 1]) >= 2:
            method = settings['method']
            dims = settings['dims']

            if method == 'newton-schulz': X = zeropower_via_newtonschulz5(target, settings['ns_steps'], dims=dims)
            elif method == 'svd': X = _svd_orthogonalize_(target, dims=dims)
            else: raise ValueError(method)

            if settings['adaptive']:
                assert grad is not None
                X = _adaptive_scaling(X, grad, dims)

            return X.view_as(target)
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


