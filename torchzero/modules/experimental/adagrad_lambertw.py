from operator import itemgetter
from typing import Literal

import torch
import scipy.special
import numpy as np
from ...core import (
    Chainable,
    Module,
    Target,
    TensorwiseTransform,
    Transform,
    Var,
    apply_transform,
)
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states
from ...utils.linalg import matrix_power_eigh
from ..functional import add_power_, lerp_power_, root


def _vpv(v: torch.Tensor) -> torch.Tensor:
    """ calculate X where X_ij is v_i^v_j"""
    return v.unsqueeze(1) ** v

def _matrix_lambertw_Ax(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    L, Q = torch.linalg.eigh(A)
    L.clip_(min = -1/torch.e)

    W_L_np = scipy.special.lambertw(L.numpy(force=True), k=0).real
    W_L = torch.from_numpy(W_L_np).to(A)

    v_prime = Q.T @ v
    y = v_prime / W_L.clip(min=1e-4)
    x = Q @ y
    return x

def _lambertw_newton_raphson(x: torch.Tensor, iterations=5):
    z = (x+1).log_()
    for _ in range(iterations):
        exp_z = z.exp()
        numerator = z * exp_z - x
        denominator = exp_z * (z + 1.0) + 1e-8
        delta = numerator / denominator
        z -= delta
    return z

class AdagradLambertW(TensorwiseTransform):
    """diabolical"""
    def __init__(self, beta: float | None = None, decay: float | None = None, concat_params=True, update_freq=1, init: Literal['identity', 'zeros', 'ones', 'GGT'] = 'identity', divide: bool=True, inner: Chainable | None = None):
        defaults = dict(beta=beta, decay=decay, init=init, divide=divide)
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, loss, state, settings):
        g = tensor.ravel().abs().clip_(max=20)
        GpG = _vpv(g)
        decay = settings['decay']
        beta = settings['beta']
        init = settings['init']

        if 'GG' not in state:
            if init == 'identity': state['GG'] = torch.eye(GpG.size(0), device=GpG.device, dtype=GpG.dtype)
            elif init == 'zeros': state['GG'] =  torch.zeros_like(GpG)
            elif init == 'ones': state['GG'] = torch.ones_like(GpG)
            elif init == 'GGT': state['GG'] = GpG.clone()
            else: raise ValueError(init)
        if decay is not None: state['GG'].mul_(decay)

        if beta is not None: state['GG'].lerp_(GpG, 1-beta)
        else: state['GG'].add_(GpG)
        state['i'] = state.get('i', 0) + 1 # number of GGTs in sum

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, loss, state, settings):
        GpG = state['GG']
        divide = settings['divide']
        if divide: GpG = GpG/state.get('i', 1)

        if tensor.numel() == 1:
            GpG = GpG.squeeze()
            return tensor / _lambertw_newton_raphson(GpG).clip_(min=1e-8)

        try:
            return _matrix_lambertw_Ax(GpG, tensor.ravel()).view_as(tensor)

        except torch.linalg.LinAlgError:
            scale = 1 / tensor.abs().max()
            return tensor.mul_(scale.clip(min=torch.finfo(tensor.dtype).eps, max=1)) # conservative scaling

