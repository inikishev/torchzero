from collections.abc import Callable, Iterable

import torch
import torch_dct

from .fft_space import RFFTWrapper

def reverse_dims(t:torch.Tensor):
    return t.permute(*reversed(range(t.ndim)))

class DCTWrapper(RFFTWrapper):
    def __init__(
        self,
        params,
        optimizer_fn: Callable[[list[torch.Tensor]], torch.optim.Optimizer],
        dims: int = 3,
        norm=None,
    ):
        super().__init__(params, optimizer_fn)
        self.dims = dims
        self.norm = norm

    def project(self, grads: list[torch.Tensor]) -> list[torch.Tensor]:

        projected = []
        for g in grads:
            g = reverse_dims(g)
            dim = min(g.ndim, self.dims)

            if dim == 1: dct = torch_dct.dct(g, norm = self.norm)
            elif dim == 2: dct = torch_dct.dct_2d(g, norm=self.norm)
            elif dim == 3: dct = torch_dct.dct_3d(g, norm=self.norm)
            else: raise ValueError(f"Unsupported number of dimensions {dim}")

            projected.append(reverse_dims(dct))

        return projected

    def unproject(self, update: list[torch.Tensor]) -> list[torch.Tensor]:

        unprojected = []
        for g in update:
            g = reverse_dims(g)
            dim = min(g.ndim, self.dims)

            if dim == 1: dct = torch_dct.idct(g, norm = self.norm)
            elif dim == 2: dct = torch_dct.idct_2d(g, norm=self.norm)
            elif dim == 3: dct = torch_dct.idct_3d(g, norm=self.norm)
            else: raise ValueError(f"Unsupported number of dimensions {dim}")

            unprojected.append(reverse_dims(dct))

        return unprojected
