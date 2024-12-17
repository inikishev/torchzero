import typing as T
from collections import abc

import torch

from ...tensorlist import Distributions, TensorList
from ...core import OptimizerModule


class Dropout(OptimizerModule):
    def __init__(self, p: float = 0.5):
        """
        Applies dropout to the update - sets random elements to 0.

        This can be used to apply learning rate dropout, if put after other modules, or gradient dropout,
        if put first.

        Args:
            p (float, optional): probability to replace update value with zero. Defaults to 0.5.

        Reference:
            *Lin, H., Zeng, W., Zhuang, Y., Ding, X., Huang, Y., & Paisley, J. (2022). Learning rate dropout. IEEE Transactions on Neural Networks and Learning Systems, 34(11), 9029-9039.*
        """
        defaults = dict(p = p)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent):
        p = self.get_group_key('p')

        ascent *= ascent.bernoulli_like(p)
        return ascent
