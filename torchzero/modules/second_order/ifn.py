import warnings
from collections.abc import Callable
from functools import partial
from typing import Literal

import torch

from ...core import Chainable, Module, apply_transform, Var, HessianMethod
from ...utils import TensorList, vec_to_tensors
from ...utils.linalg.linear_operator import DenseWithInverse, Dense


class InverseFreeNewton(Module):
    """Inverse-free newton's method

    Reference
        [Massalski, Marcin, and Magdalena Nockowska-Rosiak. "INVERSE-FREE NEWTON'S METHOD." Journal of Applied Analysis & Computation 15.4 (2025): 2238-2257.](https://www.jaac-online.com/article/doi/10.11948/20240428)
    """
    def __init__(
        self,
        update_freq: int = 1,
        hessian_method: HessianMethod = "autograd",
        h: float = 1e-3,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner']
        super().__init__(defaults)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def update(self, var):
        update_freq = self.defaults['update_freq']

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        if step % update_freq == 0:
            _, _, H = var.hessian(
                hessian_method=self.defaults['hessian_method'],
                h=self.defaults['h'],
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


    def apply(self, var):
        Y = self.global_state["Y"]
        params = var.params

        # -------------------------------- inner step -------------------------------- #
        update = var.get_update()
        if 'inner' in self.children:
            update = apply_transform(self.children['inner'], update, params=params, grads=var.grad, var=var)

        g = torch.cat([t.ravel() for t in update])

        # ----------------------------------- solve ---------------------------------- #
        var.update = vec_to_tensors(Y@g, params)

        return var

    def get_H(self,var):
        return DenseWithInverse(A = self.global_state["H"], A_inv=self.global_state["Y"])