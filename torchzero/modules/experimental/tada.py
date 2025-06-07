from collections import deque

import torch

from ...core import Chainable, TensorwiseTransform
from ...utils.linalg import matrix_power_eigh


class TAda(TensorwiseTransform):
    """3rd order whitening (maybe normalizes skewness). Please note that this is experimental and isn't guaranteed to work."""
    def __init__(self, history_size: int = 100, reg: float = 1e-8, update_freq: int = 1, concat_params: bool = True, inner: Chainable | None = None):
        defaults = dict(history_size=history_size, reg=reg)
        super().__init__(defaults, uses_grad=False, update_freq=update_freq, inner=inner, concat_params=concat_params)

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, loss, state, settings):
        reg = settings['reg']
        if 'history' not in state:
            state['history'] = deque(maxlen=settings['history_size'])

        g = tensor.view(-1)
        history = state['history']
        history.append(g.clone())

        I = torch.eye(tensor.numel(), device=tensor.device, dtype=tensor.dtype).mul_(reg)
        g_k = history[0]
        outer = torch.outer(g_k, g_k).mul_(torch.dot(g_k, g).clip(min=reg))
        if len(history) > 1:
            for g_k in list(history)[1:]:
                outer += torch.outer(g_k, g_k).mul_(torch.dot(g_k, g).clip(min=reg))

        state['outer'] = outer.add_(I)

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, loss, state, settings):
        outer = state['outer']
        P = matrix_power_eigh(outer, -1/2)
        return (P @ tensor.ravel()).view_as(tensor)