from typing import Literal
from collections.abc import Iterable

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule


def vector_laplacian_smoothing(input: torch.Tensor, sigma: float = 1) -> torch.Tensor:
    """Returns a new vector with laplacian smoothing applied to it. This flattens the input!"""
    vec = input.view(-1)
    v = torch.zeros_like(vec)
    v[0] = -2
    v[1] = 1
    v[-1] = 1
    numerator = torch.fft.fft(vec) # pylint: disable = not-callable
    denominator = 1 - sigma * torch.fft.fft(v) # pylint: disable = not-callable
    return torch.fft.ifft(numerator / denominator).real # pylint: disable = not-callable

def gradient_laplacian_smoothing_(params: Iterable[torch.Tensor], sigma: float = 1, layerwise=True, min_numel = 4):
    """Applies laplacian smoothing to gradients of an iterable of parameters.

    This updates gradients in-place.

    Args:
        params (abc.Iterable[torch.Tensor]): an iterable of Tensors that will have gradients smoothed.
        sigma (float, optional): controls the amount of smoothing. Defaults to 1.
        layerwise (bool, optional):
            If True, applies smoothing to each parameter's gradient separately,
            Otherwise applies it to all gradients, concatenated into a single vector. Defaults to True.
        min_numel (int, optional):
            minimum number of elements in a parameter to apply laplacian smoothing to.
            Only has effect if `layerwise` is True. Defaults to 4.

    Reference:
        *Osher, S., Wang, B., Yin, P., Luo, X., Barekat, F., Pham, M., & Lin, A. (2022).
        Laplacian smoothing gradient descent. Research in the Mathematical Sciences, 9(3), 55.*
    """
    grads = TensorList(params).get_existing_grads()
    if layerwise:
        for g in grads:
            if g.numel() >= min_numel:
                g.set_(vector_laplacian_smoothing(g, sigma).reshape(g.shape)) # type:ignore
    else:
        vec = grads.to_vec()
        grads.from_vec_(vector_laplacian_smoothing(vec, sigma))


def _precompute_denominator(tensor: torch.Tensor, sigma) -> torch.Tensor:
    """Denominator will always be the same and depends on the size of the vector and the sigma."""
    v = torch.zeros_like(tensor.view(-1))
    v[0] = -2
    v[1] = 1
    v[-1] = 1
    return 1 - sigma * torch.fft.fft(v) # pylint: disable = not-callable

class LaplacianSmoothing(OptimizerModule):
    """Applies laplacian smoothing via a fast Fourier transform solver.

    Args:
        sigma (float, optional): controls the amount of smoothing. Defaults to 1.
        layerwise (bool, optional):
            If True, applies smoothing to each parameter's gradient separately,
            Otherwise applies it to all gradients, concatenated into a single vector. Defaults to True.
        min_numel (int, optional):
            minimum number of elements in a parameter to apply laplacian smoothing to.
            Only has effect if `layerwise` is True. Defaults to 4.
        target (str, optional):
            determines what this module updates.

            "ascent" - it updates the ascent (default).

            "grad" - it updates the gradient (and sets `.grad` attributes to updated gradient).

            "closure" - it makes a new closure that sets the updated ascent to the .`grad` attributes.

    Reference:
        *Osher, S., Wang, B., Yin, P., Luo, X., Barekat, F., Pham, M., & Lin, A. (2022).
        Laplacian smoothing gradient descent. Research in the Mathematical Sciences, 9(3), 55.*

    """
    def __init__(self, sigma:float = 1, layerwise=True, min_numel = 4, target: Literal['ascent', 'grad', 'closure',] = 'ascent'):
        # sigma from defaults is used in layerwise case
        # otherwise self.sigma is used
        defaults = dict(sigma = sigma)
        self.sigma = 1
        super().__init__(defaults, target=target)
        self.layerwise = layerwise
        self.min_numel = min_numel

        # precomputed denominator for when layerwise=False
        self.full_denominator = None


    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()
        sigmas = self.get_group_key('sigma')

        # layerwise laplacian smoothing
        if self.layerwise:

            # precompute the denominator for each layer and store it in each parameters state
            denominators = TensorList()
            for p, σ in zip(params, sigmas):
                if p.numel() > self.min_numel:
                    den = self.state[p]
                    if 'denominator' not in den: den['denominator'] = _precompute_denominator(p, σ)
                    denominators.append(den['denominator'])

            # apply the smoothing
            smoothed_direction = TensorList()
            for g, σ, den in zip(ascent, sigmas, denominators):
                smoothed_direction.append(torch.fft.ifft(torch.fft.fft(g.view(-1)) / den).real.reshape(g.shape)) # pylint: disable = not-callable
            return smoothed_direction

        # else
        # full laplacian smoothing
        # precompute full denominator
        if self.full_denominator is None:
            self.full_denominator = _precompute_denominator(ascent.to_vec(), self.sigma)

        # apply the smoothing
        vec = ascent.to_vec()
        return ascent.from_vec(torch.fft.ifft(torch.fft.fft(vec) / self.full_denominator).real) # pylint: disable = not-callable
