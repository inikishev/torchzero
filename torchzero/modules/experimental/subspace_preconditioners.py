from collections import deque

import torch
# import visualbench as vb

# import torchzero as tz

from ...core import Transform, Chainable, apply
from ...utils.linalg import inv_sqrt_2x2, matrix_power_eigh
from ...utils import TensorList, vec_to_tensors_


def inverse_sqrt(M):
    if M.shape[-1] == 2: return inv_sqrt_2x2(M, force_pd=True) # general formula for 2x2 matrices
    return matrix_power_eigh(M, -1/2)

def update_subspace_preconditioner_(
    grad: torch.Tensor, # store grads and basis as vectors for matmul
    basis: torch.Tensor, # ndim, k
    accumulator_: torch.Tensor, # k, k
    beta: float | None,
):
    projected = basis.T @ grad # k
    outer = torch.outer(projected, projected)

    if beta is None: accumulator_.add_(outer)
    else: accumulator_.lerp_(outer, 1-beta)

def apply_subspace_preconditioner(
    tensor: torch.Tensor,
    basis: torch.Tensor, # ndim, k
    accumulator: torch.Tensor,
):
    preconditioner = inverse_sqrt(accumulator) # k,k

    tensor_projected = basis.T @ tensor # k
    update_projected = preconditioner @ tensor_projected # k
    return basis @ update_projected # d

class RandomSubspacePreconditioning(Transform):
    """full matrix rmsprop in random subspace"""
    def __init__(self, k: int, beta: float | None = 0.99):
        defaults = dict(k=k, beta=beta)
        super().__init__(defaults, uses_grad=False)

    def transform(self, tensors, params, grads, vars):
        settings = self.settings[params[0]]
        g = torch.cat([t.view(-1) for t in tensors])
        k = settings['k']
        beta = settings['beta']

        if 'basis' not in self.global_state:
            self.global_state['basis'] = torch.randn(g.numel(), k, device=g.device, dtype=g.dtype)
            self.global_state['accumulator'] = torch.eye(k, device=g.device, dtype=g.dtype)

        basis = self.global_state['basis']
        accumulator = self.global_state['accumulator']

        update_subspace_preconditioner_(g, basis, accumulator, beta)
        try:
            preconditioned = apply_subspace_preconditioner(g, basis, accumulator)
        except torch.linalg.LinAlgError:
            denom = g.abs().sum()
            if denom <= 1e-10: denom = torch.ones_like(denom)
            preconditioned = g / g.abs().sum()
        vec_to_tensors_(preconditioned, tensors)

        return tensors


class HistorySubspacePreconditioning(Transform):
    """full matrix rmsprop in subspace spanned by gradient

    basis_beta is how much basis is allowed to change
    """
    def __init__(self, k: int, beta: float | None = 0.99, basis_beta=0.99, inner: Chainable | None = None):
        defaults = dict(k=k, beta=beta, basis_beta=basis_beta)
        super().__init__(defaults, uses_grad=False)

        if inner is not None: self.set_child('inner', inner)

    def transform(self, tensors, params, grads, vars):
        settings = self.settings[params[0]]

        g = torch.cat([t.view(-1) for t in tensors])
        k = settings['k']
        beta = settings['beta']
        basis_beta = settings['basis_beta']

        if 'history' not in self.global_state:
            self.global_state['history'] = deque(maxlen=k)
            self.global_state['accumulator'] = torch.eye(k, device=g.device, dtype=g.dtype)
            self.global_state['basis'] = torch.ones(g.numel(), k, device=g.device, dtype=g.dtype)


        history: deque = self.global_state['history']
        accumulator = self.global_state['accumulator']
        basis = self.global_state['basis']

        history.append(g)
        if len(history) < k:
            basis_t = torch.randn(g.numel(), k, device=g.device, dtype=g.dtype)
            history_basis = torch.stack(tuple(history), -1)
            basis_t[:, -len(history):] = history_basis

        else:
            basis_t = torch.stack(tuple(history), -1)

        basis_t[:,:-1] = basis_t[:, :-1] - basis_t[:, 1:]
        basis_t = (basis_t - basis_t.mean()) / basis_t.std()

        basis.lerp_(basis_t, 1-basis_beta)
        update_subspace_preconditioner_(g, basis, accumulator, beta)

        if 'inner' in self.children:
            tensors = apply(self.children['inner'], tensors, params, grads, vars)
            g = torch.cat([t.view(-1) for t in tensors])

        try:
            preconditioned = apply_subspace_preconditioner(g, basis, accumulator)
        except torch.linalg.LinAlgError:
            denom = g.abs().sum()
            if denom <= 1e-10: denom = torch.ones_like(denom)
            preconditioned = g / g.abs().sum()
        vec_to_tensors_(preconditioned, tensors)

        return tensors