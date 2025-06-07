from typing import Literal
import torch
import torch_dct
from .projection import Projection
from ...core import Chainable

def reverse_dims(t:torch.Tensor):
    return t.permute(*reversed(range(t.ndim)))

class DCTProjection(Projection):
    # norm description copied from pytorch docstring
    """Project update into Discrete Cosine Transform space, requires `torch_dct` library.

    Args:
        modules (Chainable): modules that will optimize the projected update.
        dims (1, 2 or 3, optional):
            applies DCT to first 1,2 or 3 dims, defaults to 3.
        norm (str, optional):
            Normalization mode.
            * None - no normalization
            * "ortho" - normalize by 1/sqrt(n)
    """

    def __init__(
        self,
        modules: Chainable,
        dims: Literal[1, 2, 3] = 3,
        norm=None,
        project_update=True,
        project_params=False,
        project_grad=False,
    ):
        defaults = dict(dims=dims, norm=norm)
        super().__init__(modules, project_update=project_update, project_params=project_params, project_grad=project_grad, defaults=defaults)

    @torch.no_grad
    def project(self, tensors, var, current):
        settings = self.settings[var.params[0]]
        dims = settings['dims']
        norm = settings['norm']

        projected = []
        for u in tensors:
            u = reverse_dims(u)
            dim = min(u.ndim, dims)

            if dim == 1: dct = torch_dct.dct(u, norm = norm)
            elif dim == 2: dct = torch_dct.dct_2d(u, norm=norm)
            elif dim == 3: dct = torch_dct.dct_3d(u, norm=norm)
            else: raise ValueError(f"Unsupported number of dimensions {dim}")

            projected.append(dct)

        return projected

    @torch.no_grad
    def unproject(self, tensors, var, current):
        settings = self.settings[var.params[0]]
        dims = settings['dims']
        norm = settings['norm']

        unprojected = []
        for u in tensors:
            dim = min(u.ndim, dims)

            if dim == 1: idct = torch_dct.idct(u, norm = norm)
            elif dim == 2: idct = torch_dct.idct_2d(u, norm=norm)
            elif dim == 3: idct = torch_dct.idct_3d(u, norm=norm)
            else: raise ValueError(f"Unsupported number of dimensions {dim}")

            unprojected.append(reverse_dims(idct))

        return unprojected
