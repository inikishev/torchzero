import typing as T
from collections import abc

import torch

from ...tensorlist import Distributions, TensorList
from ...core import OptimizerModule


class AddNoise(OptimizerModule):
    def __init__(self, alpha: float = 1e-2, distribution: Distributions = 'normal'):
        """Add noise to update. Note that noise ignores magnitude of the update.

        Args:
            alpha (float, optional): magnitude of noise. Defaults to 1e-2.
            distribution (Distributions, optional): distribution of noise. Defaults to 'normal'.
        """
        defaults = dict(alpha = alpha)
        super().__init__(defaults)
        self.distribution: Distributions = distribution

    @torch.no_grad
    def _update(self, state, ascent):
        alpha = self.get_group_key('alpha')

        ascent += ascent.sample_like(alpha, self.distribution)
        return ascent