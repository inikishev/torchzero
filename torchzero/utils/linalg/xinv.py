import torch

def x_inv(
    diag: torch.Tensor,
    antidiag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """invert an x-matrix with no checks that it is invertible"""
    n = diag.shape[0]
    if diag.dim() != 1 or antidiag.dim() != 1 or antidiag.shape[0] != n:
        raise ValueError("Input tensors must be 1D and have the same size.")
    if n == 0:
        return torch.empty_like(diag), torch.empty_like(antidiag)

    # opposite indexes
    diag_rev = torch.flip(diag, dims=[0])
    antidiag_rev = torch.flip(antidiag, dims=[0])

    # determinants
    # det_i = d[i] * d[n-1-i] - a[i] * a[n-1-i]
    determinant_vec = diag * diag_rev - antidiag * antidiag_rev

    # inverse diagonal elements: y_d[i] = d[n-1-i] / det_i
    inv_diag_vec = diag_rev / determinant_vec

    # inverse anti-diagonal elements: y_a[i] = -a[i] / det_i
    inv_anti_diag_vec = -antidiag / determinant_vec

    return inv_diag_vec, inv_anti_diag_vec