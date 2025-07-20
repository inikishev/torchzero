from typing import Literal

import torch

from ...core import Chainable
from ..quasi_newton.quasi_newton import HessianUpdateStrategy, _safe_clip, bfgs_B_
from ...utils.linalg import linear_operator

def dc_bfgs_B_(B_p:torch.Tensor, B_m: torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    sy = s.dot(y)
    ss = s.dot(s)
    if sy >= 0: tau = 0
    else: tau = -sy / _safe_clip(ss) + 1e-10

    if tau == 0:
        B_p = bfgs_B_(B_p, s, y, tol)

    else:
        y_p = y + tau*s
        y_m = tau * s
        B_p = bfgs_B_(B_p, s, y_p, tol)
        B_m = bfgs_B_(B_m, s, y_m, tol)

    return B_p, B_m

class DCBFGS(HessianUpdateStrategy):
    """EXPERIMENTAL uses difference of two bees. What it does. It maintains B+ which is PD component, and B- which is concave component. What do you do with that. Idk. Maybe use trust region? Its just BFGS but without forcing hessian approximation to be PD. And yes this is compatible with trust region.
    """
    def __init__(
        self,
        alpha: float = 1,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-8,
        ptol: float | None = 1e-10,
        ptol_reset: bool = False,
        gtol: float | None = 1e-10,
        reset_interval: int | None | Literal['auto'] = None,
        beta: float | None = None,
        update_freq: int = 1,
        scale_first: bool = True,
        scale_second: bool = False,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        super().__init__(
            defaults=dict(alpha=alpha),
            init_scale=init_scale,
            tol=tol,
            ptol=ptol,
            ptol_reset=ptol_reset,
            gtol=gtol,
            reset_interval=reset_interval,
            beta=beta,
            update_freq=update_freq,
            scale_first=scale_first,
            scale_second=scale_second,
            concat_params=concat_params,
            inverse=False,
            inner=inner,
        )

    def update_B(self, B, s, y, p, g, p_prev, g_prev, state, setting):
        if 'B_m' not in state: state['B_m'] = torch.zeros_like(B)
        B_m = state['B_m']

        B, B_m = dc_bfgs_B_(B_p=B, B_m=B_m, s=s, y=y, tol=setting['tol'])
        state['B_m'] = B_m
        return B

    def modify_B(self, B, state, setting):
        if "B_m" not in state: return B
        alpha = setting['alpha']
        return (B - state["B_m"]*alpha)

    def get_B(self, var):
        param = var.params[0]
        state = self.state[param]
        settings = self.settings[param]
        return linear_operator.Dense(self.modify_B(state["B"], state, settings))
