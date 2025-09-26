import torch

from ...core import Chainable, Transform, HessianMethod
from ...utils import TensorList, vec_to_tensors
from ...linalg.linear_operator import DenseWithInverse


class InverseFreeNewton(Transform):
    """Inverse-free newton's method

    Reference
        [Massalski, Marcin, and Magdalena Nockowska-Rosiak. "INVERSE-FREE NEWTON'S METHOD." Journal of Applied Analysis & Computation 15.4 (2025): 2238-2257.](https://www.jaac-online.com/article/doi/10.11948/20240428)
    """
    def __init__(
        self,
        update_freq: int = 1,
        hessian_method: HessianMethod = "batched_autograd",
        h: float = 1e-3,
        inner: Chainable | None = None,
    ):
        defaults = dict(hessian_method=hessian_method, h=h)
        super().__init__(defaults, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        fs = settings[0]

        _, _, H = objective.hessian(
            hessian_method=fs['hessian_method'],
            h=fs['h'],
            at_x0=True
        )

        self.global_state["H"] = H

        # inverse free part
        if 'Y' not in self.global_state:
            num = H.T
            denom = (torch.linalg.norm(H, 1) * torch.linalg.norm(H, float('inf'))) # pylint:disable=not-callable

            finfo = torch.finfo(H.dtype)
            self.global_state['Y'] = num.div_(denom.clip(min=finfo.tiny * 2, max=finfo.max / 2))

        else:
            Y = self.global_state['Y']
            I2 = torch.eye(Y.size(0), device=Y.device, dtype=Y.dtype).mul_(2)
            I2 -= H @ Y
            self.global_state['Y'] = Y @ I2


    def apply_states(self, objective, states, settings):
        Y = self.global_state["Y"]
        g = torch.cat([t.ravel() for t in objective.get_updates()])
        objective.updates = vec_to_tensors(Y@g, objective.params)
        return objective

    def get_H(self,objective=...):
        return DenseWithInverse(A = self.global_state["H"], A_inv=self.global_state["Y"])