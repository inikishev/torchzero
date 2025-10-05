from collections import deque
from typing import Literal

import torch

from torchzero.core import Chainable, TensorTransform
from torchzero.linalg import matrix_power_eigh, torch_linalg, orthogonalize, OrthogonalizeMethod, regularize_eigh
from torchzero.utils import TensorList, vec_to_tensors_


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

# yeah so I can also run subspace opts in this basis
def apply_subspace_preconditioner(
    tensor: torch.Tensor,
    basis: torch.Tensor, # ndim, k
    accumulator: torch.Tensor,
    tol: float,
    truncate: int | None,
    damping: float,
    rdamping: float,
):
    L, Q = torch_linalg.eigh(accumulator, retry_float64=True)
    L, Q = regularize_eigh(L=L, Q=Q, truncate=truncate, tol=tol, damping=damping, rdamping=rdamping)

    if L is None or Q is None:
        return tensor.clip(-0.1, 0.1)

    preconditioner = (Q * L.rsqrt().unsqueeze(-2)) @ Q.mH

    tensor_projected = basis.T @ tensor # k
    update_projected = preconditioner @ tensor_projected # k
    return basis @ update_projected # d


class CommonDirectionsWhiten(TensorTransform):
    """Whitens in subspace spanned by history of gradient differences.

    Args:
        beta - for preconditioner itself in the basis.
        basis_beta - how much basis is allowed to change.
    """

    def __init__(
        self,
        k: int = 100,
        beta: float | None = 0.95,
        basis_beta=0.95,
        tol: float = 1e-7,
        truncate: int | None = None,
        damping: float = 1e-4,
        rdamping: float = 0,
        basis_type: Literal["gradients", "differences"] = "differences",
        orthogonalize_method: OrthogonalizeMethod | None = 'newtonschulz',

        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        for key in ["self", "inner", "concat_params"]:
            del defaults[key]

        super().__init__(defaults, concat_params=concat_params, inner=inner)

    @torch.no_grad
    def single_tensor_update(self, tensor, param, grad, loss, state, setting):
        g = tensor.ravel()
        k = setting['k']
        beta = setting['beta']
        basis_beta = setting['basis_beta']
        step = state.get("step", 0)
        state["step"] = step + 1

        # initialize history
        if 'history' not in state:
            state['history'] = deque(maxlen=k)
            state['accumulator'] = torch.eye(k, device=g.device, dtype=g.dtype)
            state['basis'] = torch.zeros(g.numel(), k, device=g.device, dtype=g.dtype)

        history: deque = state['history']
        accumulator = state['accumulator']
        basis = state['basis']
        history.append(g)

        # stack history to new basis term, if history isn't full, fill with random vecs
        if len(history) < k:
            basis_t = torch.randn(g.numel(), k, device=g.device, dtype=g.dtype)
            history_basis = torch.stack(tuple(history), -1)
            basis_t[:, -len(history):] = history_basis

        else:
            basis_t = torch.stack(tuple(history), -1)

        # in this case basis uses differences in gradients except last entry is the gradient
        if setting["basis_type"] == "differences":
            basis_t[:,:-1] = basis_t[:, :-1] - basis_t[:, 1:]

        # normalize or orthonormalize new basis term
        if setting["orthogonalize_method"] is not None:
            basis_t = orthogonalize(basis_t, method = setting["orthogonalize_method"])
        else:
            basis_t = (basis_t - basis_t.mean()) / basis_t.std().clip(min=torch.finfo(g.dtype).tiny * 2)

        # lerp basis
        basis.lerp_(basis_t, 1-basis_beta)
        basis = basis /  (1 - basis_beta ** (step+1)) # correct bias on basis EMA
        update_subspace_preconditioner_(g, basis, accumulator, beta)

    @torch.no_grad
    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        g = tensor.ravel()

        basis = state['basis']
        accumulator = state['accumulator']
        step = state["step"]
        accumulator = accumulator / (1 - setting["beta"] ** (step+1)) # correct bias on accumulator EMA

        try:
            preconditioned = apply_subspace_preconditioner(
                g,
                basis,
                accumulator,
                tol=setting["tol"],
                truncate=setting["truncate"],
                damping=setting["damping"],
                rdamping=setting["rdamping"],
            )
        except torch.linalg.LinAlgError:
            preconditioned = g.clip(-0.1, 0.1)

        return preconditioned.view_as(tensor)

