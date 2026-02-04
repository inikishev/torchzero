from collections.abc import Callable

import torch

from ...core import Chainable, Transform, HessianMethod
from ...utils import TensorList, vec_to_tensors_, unpack_states
from ..opt_utils import safe_clip
from .newton import _newton_update_state_, _newton_solve, _newton_get_H

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
        eigval_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        eigv_tol: float | None = None,
        truncate: int | None = None,
        update_freq: int = 1,
        precompute_inverse: bool | None = None,
        use_lstsq: bool = False,
        hessian_method: HessianMethod = "batched_autograd",
        h: float = 1e-3,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner'], defaults["update_freq"]
        super().__init__(defaults, update_freq=update_freq, inner=inner, )

    @torch.no_grad
    def update_states(self, objective, states, settings):
        fs = settings[0]

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
        if "H" not in self.global_state:
            x_prev.copy_(x_list)
            f_prev.copy_(f_list)
            P = J

        # INM update
        else:
            s_list = x_list - x_prev
            y_list = f_list - f_prev
            x_prev.copy_(x_list)
            f_prev.copy_(f_list)

            P = inm(f, J, s=s_list.to_vec(), y=y_list.to_vec())

        # update state
        precompute_inverse = fs["precompute_inverse"]
        if precompute_inverse is None:
            precompute_inverse = fs["__update_freq"] >= 10

        _newton_update_state_(
            H=P,
            state = self.global_state,
            damping = fs["damping"],
            eigval_fn = fs["eigval_fn"],
            eigv_tol = fs["eigv_tol"],
            truncate = fs["truncate"],
            precompute_inverse = precompute_inverse,
            use_lstsq = fs["use_lstsq"]
        )

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        updates = objective.get_updates()
        fs = settings[0]

        b = torch.cat([t.ravel() for t in updates])
        sol = _newton_solve(b=b, state=self.global_state, use_lstsq=fs["use_lstsq"])

        vec_to_tensors_(sol, updates)
        return objective


    def get_H(self,objective=...):
        return _newton_get_H(self.global_state)
