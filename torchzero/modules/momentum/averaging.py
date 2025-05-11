from collections import deque
from collections.abc import Sequence
from typing import Any, Literal, cast

import torch

from ...core import TensorwiseTransform, Target
from ...utils import tolist


class Averaging(TensorwiseTransform):
    def __init__(self, history_size: int, target: Target = 'update'):
        defaults = dict(history_size=history_size)
        super().__init__(uses_grad=False, defaults=defaults, target=target)

    @torch.no_grad
    def transform(self, tensor, param, grad, vars):
        history_size = self.settings[param]['history_size']
        state = self.state[param]
        if 'history' not in state:
            state['history'] = deque(maxlen=history_size)
            state['average'] = torch.zeros_like(tensor)

        history = state['history']; average = state['average']
        if len(history) == history_size: average -= history[0]
        history.append(tensor)
        average += tensor

        return average / len(history)

class WeightedAveraging(TensorwiseTransform):
    def __init__(self, weights: Sequence[float] | torch.Tensor | Any, target: Target = 'update'):
        defaults = dict(weights = tolist(weights))
        super().__init__(uses_grad=False, defaults=defaults, target=target)

    @torch.no_grad
    def transform(self, tensor, param, grad, vars):
        weights = self.settings[param]['weights']
        state = self.state[param]

        if 'history' not in state:
            state['history'] = deque(maxlen=len(weights))

        history = state['history']
        history.append(tensor)

        average = cast(torch.Tensor, None)
        for i, (h, w) in enumerate(zip(history, weights)):
            if i == 0: average = h * (w / len(history))
            else: average += h * (w / len(history))

        return average


class MedianAveraging(TensorwiseTransform):
    def __init__(self, history_size: int, target: Target = 'update'):
        defaults = dict(history_size = history_size)
        super().__init__(uses_grad=False, defaults=defaults, target=target)

    @torch.no_grad
    def transform(self, tensor, param, grad, vars):
        history_size = self.settings[param]['history_size']
        state = self.state[param]

        if 'history' not in state:
            state['history'] = deque(maxlen=history_size)

        history = state['history']
        history.append(tensor)

        stacked = torch.stack(tuple(history), 0)
        return torch.quantile(stacked, 0.5, dim = 0)
