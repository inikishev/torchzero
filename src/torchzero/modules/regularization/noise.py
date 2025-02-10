from collections import abc
from typing import Literal

import torch

from ...core import OptimizerModule
from ...tensorlist import Distributions, TensorList, _Scalar, _ScalarSequence


def add_noise_(
    grads: abc.Iterable[torch.Tensor],
    alpha: "_Scalar | _ScalarSequence" = 1e-2,
    distribution: Distributions = "normal",
    mode: Literal["absolute", "global", "param", "channel"] = "param",
):
    if not isinstance(grads, TensorList): grads = TensorList(grads)
    if mode == 'absolute':
        grads += grads.sample_like(alpha, distribution)

    elif mode == 'global':
        grads += grads.sample_like((grads.total_vector_norm(1)/grads.total_numel() * alpha).detach().cpu().item(), distribution) # type:ignore

    elif mode == 'param':
        grads += grads.sample_like(grads.abs().mean()*alpha, distribution)

    elif mode == 'channel':
        grads = grads.unbind_channels()
        grads += grads.sample_like(grads.abs().mean()*alpha, distribution)

class AddNoise(OptimizerModule):
    """Add noise to update. By default noise magnitude is relative to the mean of each parameter.

    Args:
        alpha (float, optional): magnitude of noise. Defaults to 1e-2.
        distribution (Distributions, optional): distribution of noise. Defaults to 'normal'.
        mode (str, optional):
            how to calculate noise magnitude.

            - "absolute": ignores gradient magnitude and always uses `alpha` as magnitude.

            - "global": multiplies `alpha` by mean of the entire gradient, as if it was a single vector.

            - "param": multiplies `alpha` by mean of each individual parameter (default).

            - "channel": multiplies `alpha` by mean of each channel of each parameter.
        """

    def __init__(
        self,
        alpha: float = 1.,
        distribution: Distributions = "normal",
        mode: Literal["absolute", "global", "param", "channel"] = "param",
    ):
        defaults = dict(alpha = alpha)
        super().__init__(defaults)
        self.distribution: Distributions = distribution
        self.mode: Literal["absolute", "global", "param", "channel"] = mode

    @torch.no_grad
    def _update(self, vars, ascent):
        alpha = self.get_group_key('alpha')

        add_noise_(ascent, alpha, self.distribution, self.mode)
        return ascent

class Random(OptimizerModule):
    """uses a random vector as the update. The vector is completely random and isn't checked to be descent direction.
    This is therefore mainly useful in combination with other modules like Sum, Multiply, etc."""
    def __init__(self, alpha: float = 1, distribution: Distributions = "normal"):
        defaults = dict(alpha = alpha)
        super().__init__(defaults)
        self.distribution: Distributions = distribution

    @torch.no_grad
    def _update(self, vars, ascent):
        alpha = self.get_group_key('alpha')
        return ascent.sample_like(alpha, self.distribution)