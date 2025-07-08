from collections import deque

import torch
# import visualbench as vb

# import torchzero as tz

from ...core import Transform, Chainable, apply_transform
from ...utils.linalg import inv_sqrt_2x2, matrix_power_eigh, gram_schmidt
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
    """Whitens in random slowly changing subspace.

    .. warning::
        Experimental and this is a barebones implementation.

    """
    def __init__(self, k: int, beta: float | None = 0.99, basis_beta: float | None = 0.99, inner: Chainable | None = None):
        defaults = dict(k=k, beta=beta, basis_beta=basis_beta)
        super().__init__(defaults, uses_grad=False)

        if inner is not None: self.set_child('inner', inner)

    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        settings = settings[0]
        g = torch.cat([t.view(-1) for t in tensors])
        k = settings['k']
        beta = settings['beta']
        basis_beta = settings['basis_beta']

        if 'basis' not in self.global_state:
            self.global_state['basis'] = torch.randn(g.numel(), k, device=g.device, dtype=g.dtype)
            self.global_state['accumulator'] = torch.eye(k, device=g.device, dtype=g.dtype)

        basis = self.global_state['basis']
        accumulator = self.global_state['accumulator']

        if basis_beta is not None:
            basis.lerp_(torch.randn_like(basis), 1-basis_beta)

        update_subspace_preconditioner_(g, basis, accumulator, beta)

        if 'inner' in self.children:
            tensors = apply_transform(self.children['inner'], tensors, params, grads)
            g = torch.cat([t.view(-1) for t in tensors])

        try:
            preconditioned = apply_subspace_preconditioner(g, basis, accumulator)
        except torch.linalg.LinAlgError:
            preconditioned = g.clip(-0.1, 0.1)
        vec_to_tensors_(preconditioned, tensors)

        return tensors


class HistorySubspacePreconditioning(Transform):
    """Whitens in subspace spanned by history of gradient differences.

    .. warning::
        Experimental and this is a barebones implementation.

    Args:
        beta - for preconditioner itself in the basis.
        basis_beta - how much basis is allowed to change.
    """
    def __init__(self, k: int, beta: float | None = 0.99, basis_beta=0.99, inner: Chainable | None = None):
        defaults = dict(k=k, beta=beta, basis_beta=basis_beta)
        super().__init__(defaults, uses_grad=False)

        if inner is not None: self.set_child('inner', inner)

    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        settings = settings[0]

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
            tensors = apply_transform(self.children['inner'], tensors, params, grads)
            g = torch.cat([t.view(-1) for t in tensors])

        try:
            preconditioned = apply_subspace_preconditioner(g, basis, accumulator)
        except torch.linalg.LinAlgError:
            preconditioned = g.clip(-0.1,0.1)
        vec_to_tensors_(preconditioned, tensors)

        return tensors

