import typing
from collections import abc

import numpy as np
import torch

import nevergrad as ng

from ...core import TensorListOptimizer

class NLOptOptimizer(TensorListOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._optimizer = None

    def _get_optimizer(self, loss_fn: typing.Callable[[torch.Tensor], torch.Tensor]) -> abc.Mapping:
        if self._optimizer is None: