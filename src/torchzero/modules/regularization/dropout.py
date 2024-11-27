import typing as T
from collections import abc

import torch

from ...tensorlist import Distributions, TensorList
from ...core import OptimizerModule


class Dropout(OptimizerModule):
    def __init__(self, p: float = 0.5):
        """
        Implements learning rate dropout
        (and gradient dropout at the same time if this is applied before calculating an update).

        LR dropout randomly drops learning rates for some parameters on each step, i.e doesn't update them.

        Lin, H., Zeng, W., Zhuang, Y., Ding, X., Huang, Y., & Paisley, J. (2022). Learning rate dropout. IEEE Transactions on Neural Networks and Learning Systems, 34(11), 9029-9039.
        Args:
            p (float, optional): Dropout probability. Defaults to 0.5.
        """
        defaults = dict(p = p)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent_direction):
        p = self.get_group_key('p')

        ascent_direction *= ascent_direction.bernoulli_like(p)
        return ascent_direction
