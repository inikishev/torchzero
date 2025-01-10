import typing
from collections.abc import Mapping, Callable

import numpy as np
import torch

import nlopt
from ...core import TensorListOptimizer

class NLOptOptimizer(TensorListOptimizer):
    """i haven't made this one yet"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._optimizer = None

    def _get_optimizer(self, loss_fn: Callable[[torch.Tensor], torch.Tensor]):# -> Mapping:
        if self._optimizer is None: ...