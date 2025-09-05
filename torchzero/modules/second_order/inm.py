from collections.abc import Callable
from typing import Literal

import torch

from ...core import Chainable, Module
from ...utils import TensorList, vec_to_tensors
from ..functional import safe_clip
from .newton import _get_H, _get_loss_grad_and_hessian, _newton_step

@torch.no_grad
def inm(f:torch.Tensor, J:torch.Tensor, s:torch.Tensor, y:torch.Tensor):

    yy = safe_clip(y.dot(y))
    ss = safe_clip(s.dot(s))

    term1 = y.dot(y - J@s) / yy
    FbT = f.outer(s).mul_(term1 / ss)

    P = FbT.add_(J)
    return P

def _eigval_fn(J: torch.Tensor, fn) -> torch.Tensor:
    if fn is None: return J
    L, Q = torch.linalg.eigh(J) # pylint:disable=not-callable
    return (Q * L.unsqueeze(-2)) @ Q.mH

class INM(Module):
    """Improved Newton's Method (INM).

    Reference:
        [Saheya, B., et al. "A new Newton-like method for solving nonlinear equations." SpringerPlus 5.1 (2016): 1269.](https://d-nb.info/1112813721/34)
    """

    def __init__(
        self,
        damping: float = 0,
        use_lstsq: bool = False,
        update_freq: int = 1,
        hessian_method: Literal["autograd", "func", "autograd.functional"] = "autograd",
        vectorize: bool = True,
        inner: Chainable | None = None,
        H_tfm: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, bool]] | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        eigval_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        defaults = dict(damping=damping, hessian_method=hessian_method, use_lstsq=use_lstsq, vectorize=vectorize, H_tfm=H_tfm, eigval_fn=eigval_fn, update_freq=update_freq)
        super().__init__(defaults)

        if inner is not None:
            self.set_child("inner", inner)

    @torch.no_grad
    def update(self, var):
        update_freq = self.defaults['update_freq']

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        if step % update_freq == 0:
            _, f_list, J = _get_loss_grad_and_hessian(
                var, self.defaults['hessian_method'], self.defaults['vectorize']
            )

            f = torch.cat([t.ravel() for t in f_list])
            J = _eigval_fn(J, self.defaults["eigval_fn"])

            x_list = TensorList(var.params)
            f_list = TensorList(var.get_grad())
            x_prev, f_prev = self.get_state(var.params, "x_prev", "f_prev", cls=TensorList)

            # initialize on 1st step, do Newton step
            if step == 0:
                x_prev.copy_(x_list)
                f_prev.copy_(f_list)
                self.global_state["P"] = J
                return

            # INM update
            s_list = x_list - x_prev
            y_list = f_list - f_prev
            x_prev.copy_(x_list)
            f_prev.copy_(f_list)

            self.global_state["P"] = inm(f, J, s=s_list.to_vec(), y=y_list.to_vec())


    @torch.no_grad
    def apply(self, var):
        params = var.params
        update = _newton_step(
            var=var,
            H = self.global_state["P"],
            damping=self.defaults["damping"],
            inner=self.children.get("inner", None),
            H_tfm=self.defaults["H_tfm"],
            eigval_fn=None, # it is applied in `update`
            use_lstsq=self.defaults["use_lstsq"],
        )

        var.update = vec_to_tensors(update, params)

        return var

    def get_H(self,var=...):
        return _get_H(self.global_state["P"], eigval_fn=None)
