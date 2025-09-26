from collections.abc import Callable

import torch

from ...core import Chainable, Transform, HessianMethod
from ...utils import TensorList, vec_to_tensors, unpack_states
from ..functional import safe_clip
from .newton import _get_H, _newton_step

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

class ImprovedNewton(Transform):
    """Improved Newton's Method (INM).

    Reference:
        [Saheya, B., et al. "A new Newton-like method for solving nonlinear equations." SpringerPlus 5.1 (2016): 1269.](https://d-nb.info/1112813721/34)
    """

    def __init__(
        self,
        damping: float = 0,
        use_lstsq: bool = False,
        update_freq: int = 1,
        H_tfm: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, bool]] | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        eigval_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        hessian_method: HessianMethod = "batched_autograd",
        h: float = 1e-3,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner']
        super().__init__(defaults, inner=inner)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        fs = settings[0]
        update_freq = fs['update_freq']
        step = self.increment_counter("step", 0)

        if step % update_freq == 0:
            _, f_list, J = objective.hessian(
                hessian_method=fs['hessian_method'],
                h=fs['h'],
                at_x0=True
            )
            if f_list is None: f_list = objective.get_grads()

            f = torch.cat([t.ravel() for t in f_list])
            J = _eigval_fn(J, fs["eigval_fn"])

            x_list = TensorList(objective.params)
            f_list = TensorList(objective.get_grads())
            x_prev, f_prev = unpack_states(states, objective.params, "x_prev", "f_prev", cls=TensorList)

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
    def apply_states(self, objective, states, settings):
        fs = settings[0]

        update = _newton_step(
            objective=objective,
            H = self.global_state["P"],
            damping=fs["damping"],
            inner=None,
            H_tfm=fs["H_tfm"],
            eigval_fn=None, # it is applied in `update`
            use_lstsq=fs["use_lstsq"],
        )

        objective.updates = vec_to_tensors(update, objective.params)

        return objective

    def get_H(self,objective=...):
        return _get_H(self.global_state["P"], eigval_fn=None)
