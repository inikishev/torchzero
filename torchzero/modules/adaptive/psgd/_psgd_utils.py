# pylint:disable=not-callable
import warnings

import torch

from .psgd import lift2single


def _initialize_lra_state_(tensor: torch.Tensor, state, setting):
    n = tensor.numel()
    rank = max(min(setting["rank"], n-1), 1)
    dtype=tensor.dtype
    device=tensor.device

    U = torch.randn((n, rank), dtype=dtype, device=device)
    U *= 0.1**0.5 / torch.linalg.vector_norm(U)

    V = torch.randn((n, rank), dtype=dtype, device=device)
    V *= 0.1**0.5 / torch.linalg.vector_norm(V)

    if setting["init_scale"] is None:
        # warnings.warn("FYI: Will set the preconditioner initial scale on the fly. Recommend to set it manually.")
        d = None
    else:
        d = torch.ones(n, 1, dtype=dtype, device=device) * setting["init_scale"]

    state["UVd"] = [U, V, d]
    state["Luvd"] = [lift2single(torch.zeros([], dtype=dtype, device=device)) for _ in range(3)]



def _wrap_with_no_backward(opt):
    """to use original psgd opts with visualbench"""
    class _Wrapped:
        def step(self, closure):
            return opt.step(lambda: closure(False))
    return _Wrapped()