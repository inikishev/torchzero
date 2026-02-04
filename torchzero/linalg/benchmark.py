from collections.abc import Callable

import torch


def benchmark_solver(
    A: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    solver: Callable[[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]]
):
    residuals = []
    def A_mm(x):
        if callable(A): Ax = A(x)
        else: Ax = A@x
        residuals.append(torch.linalg.vector_norm(Ax-b)) # pylint:disable=not-callable
        return Ax

    solver(A_mm, b)
    return residuals

