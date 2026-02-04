from typing import Literal
from collections.abc import Iterable

import torch

from ...utils.tensorlist import TensorList
from ...core import TensorTransform


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
    grads = TensorList(params).get_grad()
    if layerwise:
        for g in grads:
            if g.numel() >= min_numel:
                g.set_(vector_laplacian_smoothing(g, sigma).view_as(g)) # pyright:ignore[reportArgumentType]
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

class LaplacianSmoothing(TensorTransform):
    """Applies laplacian smoothing via a fast Fourier transform solver which can improve generalization.

    Args:
        sigma (float, optional): controls the amount of smoothing. Defaults to 1.
        layerwise (bool, optional):
            If True, applies smoothing to each parameter's gradient separately,
            Otherwise applies it to all gradients, concatenated into a single vector. Defaults to True.
        min_numel (int, optional):
            minimum number of elements in a parameter to apply laplacian smoothing to.
            Only has effect if `layerwise` is True. Defaults to 4.
        target (str, optional):
            what to set on var.

    Examples:
    Laplacian Smoothing Gradient Descent optimizer as in the paper

    ```python

    opt = tz.Optimizer(
        model.parameters(),
        tz.m.LaplacianSmoothing(),
        tz.m.LR(1e-2),
    )
    ```

    Reference:
        Osher, S., Wang, B., Yin, P., Luo, X., Barekat, F., Pham, M., & Lin, A. (2022). Laplacian smoothing gradient descent. Research in the Mathematical Sciences, 9(3), 55.

    """
    def __init__(self, sigma:float = 1, layerwise=True, min_numel = 4):
        defaults = dict(sigma = sigma, layerwise=layerwise, min_numel=min_numel)
        super().__init__(defaults)
        # precomputed denominator for when layerwise=False
        self.global_state['full_denominator'] = None


    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        layerwise = settings[0]['layerwise']

        # layerwise laplacian smoothing
        if layerwise:

            # precompute the denominator for each layer and store it in each parameters state
            smoothed_target = TensorList()
            for p, t, state, setting in zip(params, tensors, states, settings):
                if p.numel() > setting['min_numel']:
                    if 'denominator' not in state: state['denominator'] = _precompute_denominator(p, setting['sigma'])
                    smoothed_target.append(torch.fft.ifft(torch.fft.fft(t.view(-1)) / state['denominator']).real.view_as(t)) #pylint:disable=not-callable
                else:
                    smoothed_target.append(t)

            return smoothed_target

        # else
        # full laplacian smoothing
        # precompute full denominator
        tensors = TensorList(tensors)
        if self.global_state.get('full_denominator', None) is None:
            self.global_state['full_denominator'] = _precompute_denominator(tensors.to_vec(), settings[0]['sigma'])

        # apply the smoothing
        vec = tensors.to_vec()
        return tensors.from_vec(torch.fft.ifft(torch.fft.fft(vec) / self.global_state['full_denominator']).real)#pylint:disable=not-callable


