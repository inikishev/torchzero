"""
Newton-Schulz iteration code is taken from https://github.com/KellerJordan/Muon

Keller Jordan and Yuchen Jin and Vlado Boza and You Jiacheng and Franz Cecista and Laker Newhouse and Jeremy Bernstein.
Muon: An optimizer for hidden layers in neural networks (2024). URL: https://kellerjordan.github.io/posts/muon
"""
from collections.abc import Iterable

import torch

from ...core import OptimizerModule
# from ...utils.compile import maybe_compile

def _zeropower_via_newtonschulz5(G, steps):
    """
    code from https://github.com/KellerJordan/Muon

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X

_compiled_zeropower_via_newtonschulz5 = torch.compile(_zeropower_via_newtonschulz5)


def zeropower_via_newtonschulz_(params: Iterable[torch.Tensor], steps: int = 6, adaptive = False, compiled = True):
    """Uses newton-Schulz iteration to compute the zeroth power / orthogonalization of gradients of an iterable of parameters.

    This sets gradients in-place.

    Note that the Muon page says that embeddings and classifier heads should not be orthogonalized.

    The orthogonalization code is taken from https://github.com/KellerJordan/Muon
    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to orthogonalize.
        steps (int): The number of Newton-Schulz iterations to run. (6 is probably always enough).
            The number of Newton-Schulz iterations to run. (6 is probably always enough). Defaults to 6.
        adaptive (bool, optional):
            Enables adaptation to scale of gradients (from https://github.com/leloykun/adaptive-muon). Defaults to False.
        compiled (bool, optional):
            Uses compiled newton-Schulz iteration function. Faster but won't work on windows. Defaults to True.


    """
    if compiled: fn = _compiled_zeropower_via_newtonschulz5
    else: fn = _zeropower_via_newtonschulz5
    for p in params:
        if p.grad is not None and p.grad.ndim >= 2 and min(p.grad.shape) >= 2:
            G = p.grad.view(p.grad.shape[0], -1)
            X = fn(G, steps)

            if adaptive:
                # this is from https://github.com/leloykun/adaptive-muon
                X = torch.einsum('ij,ij,ab->ab', G.type_as(X), X, X)  # Adaptive scaling,`(G * X).sum() * X` == (G.T @ X).trace() * X

            p.grad = X.reshape_as(p.grad).to(p.grad, copy=False)


class ZeropowerViaNewtonSchulz(OptimizerModule):
    """Uses Newton-Schulz iteration to compute the zeroth power / orthogonalization of gradients of an iterable of parameters.

    To disable orthogonalization for a parameter, put it into a parameter group with "newtonshultz" = False.
    The Muon page says that embeddings and classifier heads should not be orthogonalized.

    The orthogonalization code is taken from https://github.com/KellerJordan/Muon.

    Note that unlike this module, Muon also uses Adam for gradients that are not orthogonalized,
    so I'd still recommend using it. Maybe use `Wrap` to wrap it into a module (I will make muon
    with selectable modules to optimize non-muon params soon)

    However not using Adam, or putting Adam module after this to apply it to ALL updates, both seem
    to work quite well too.

    Args:
        ns_steps (int, optional):
            The number of Newton-Schulz iterations to run. (6 is probably always enough). Defaults to 6.
        adaptive (bool, optional):
            Enables adaptation to scale of gradients (from https://github.com/leloykun/adaptive-muon). Defaults to True.
        compiled (bool, optional):
            Uses compiled newton-Schulz iteration function. Faster but won't work on windows. Defaults to True.
    """
    def __init__(self, ns_steps = 6, adaptive = False, compiled=True):
        defaults = dict(newtonshultz = True, ns_steps=ns_steps, adaptive=adaptive)
        super().__init__(defaults)

        if compiled: self._zeropower_via_newtonschulz5 = _compiled_zeropower_via_newtonschulz5
        else: self._zeropower_via_newtonschulz5 = _zeropower_via_newtonschulz5

    def _update(self, state, ascent):
        toggle, ns_steps, adaptive = self.get_group_keys('newtonshultz', 'ns_steps', 'adaptive', cls=list)

        for asc, enable, steps, ada in zip(ascent, toggle, ns_steps, adaptive):
            if enable and len([i for i in asc.shape if i > 1]) != 0:
                G = asc.view(asc.shape[0], -1)
                X = self._zeropower_via_newtonschulz5(G, steps)

                if ada:
                    # this is from https://github.com/leloykun/adaptive-muon
                    X = torch.einsum('ij,ij,ab->ab', G.type_as(X), X, X)  # Adaptive scaling,`(G * X).sum() * X` == (G.T @ X).trace() * X

                asc.set_(X.reshape_as(asc).to(asc, copy=False)) # type:ignore

        return ascent



class DualNormCorrection(OptimizerModule):
    """Dual norm correction from https://github.com/leloykun/adaptive-muon.

    Description from the page:

    Single-line modification to any (dualizer-based) optimizer that allows the optimizer to adapt to the scale of the gradients as they change during training.
    This is done by scaling the dualized gradient by the clipped dual norm of the original gradient.
    """
    def __init__(self, adaptive_scale_min: int | None = -1, adaptive_scale_max: int | None = 1):
        defaults = dict(adaptive_scale_min = adaptive_scale_min, adaptive_scale_max = adaptive_scale_max)
        super().__init__(defaults)

    def _update(self, state, ascent):
        params = self.get_params()
        adaptive_scale_min, adaptive_scale_max = self.get_group_keys('adaptive_scale_min', 'adaptive_scale_max')

        for asc, grad, min, max in zip(ascent, state.maybe_compute_grad_(params), adaptive_scale_min, adaptive_scale_max):
            if len([i for i in asc.shape if i > 1]) != 0:
                scale = torch.einsum('ij,ij->', grad.view(grad.shape[0], -1), asc.view(asc.shape[0], -1))
                if min is not None or max is not None: scale = scale.clip(min, max)
                asc *= scale

        return ascent