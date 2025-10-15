from typing import Any, Literal

import torch

from ...core import TensorTransform
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states
from ..adaptive.lre_optimizers import LREOptimizerBase, _squared_reproject


def signed_cbrt(x: TensorList | Any) -> Any:
    return x.sign() * x.abs().pow(1/3)

def _clip_min_magnitude(x: torch.Tensor, eps: float):
    return x.sign() * x.abs().clamp(min=eps)

_cubic_adam_mode = Literal["signed_cbrt", "unsigned_cbrt", "halve"]

def _cubic_minimize(A: torch.Tensor | Any, B: torch.Tensor | Any, C: torch.Tensor | Any, eps):
    """minimizes (A/3)x^3 + (A/2)x^2 + Cx"""
    discriminant = B**2 - 4 * A * C

    denom = _clip_min_magnitude(2 * A, eps)
    root = discriminant.clamp(min=0).sqrt_()

    x0 = (-B + root) / denom
    x1 = (-B - root) / denom

    f0 = (A/3)*x0**3 + (B/2)*x0**2 + C*x0
    f1 = (A/3)*x1**3 + (B/2)*x1**2 + C*x1

    x_star = x0.where(f0 < f1, x1)

    adam = -C / (B + eps)
    return adam.where(discriminant < 0, x_star)

def cubic_adam_(
    tensors: TensorList,
    exp_avg_: TensorList,
    exp_avg_sq_: TensorList,
    exp_avg_cu_: TensorList,
    alpha: float | NumberList,
    beta1: float | NumberList,
    beta2: float | NumberList,
    beta3: float | NumberList,
    eps: float | NumberList,
    debiased: bool,
    step: int,

    mode: _cubic_adam_mode = 'signed_cbrt'
):
    exp_avg_.lerp_(tensors, 1-beta1)
    exp_avg_sq_.lerp_(tensors**2, 1-beta2)
    exp_avg_cu_.lerp_(tensors**3, 1-beta3)

    if debiased:
        m1 = exp_avg_ / (1 - beta1 ** step)
        m2 = exp_avg_sq_ / (1 - beta2 ** step)
        m3 = exp_avg_cu_ / (1 - beta3 ** step)
    else:
        m1, m2, m3 = exp_avg_, exp_avg_sq_, exp_avg_cu_

    # adam minimizes ax^2 + bx
    # we are going to minimize ax^3 + bx^2 + cx

    if mode == "signed_cbrt": A = signed_cbrt(m3)
    elif mode == "unsigned_cbrt": A = m3.abs().pow(1/3)
    elif mode == 'halve': A = 0.5 * m3
    else: raise ValueError(mode)

    B = m2.sqrt()
    C = m1
    x_star = _cubic_minimize(A, B, C, eps)
    return x_star.mul_(-alpha)

class CubicAdam(TensorTransform):
    """Adam which has 3rd momentum and minimizes a cubic polynomial."""
    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.99,
        beta3: float = 0.99,
        eps: float = 1e-8,
        debiased:bool=True,
        alpha: float = 1.,

        mode: _cubic_adam_mode = 'signed_cbrt'
    ):
        defaults=dict(beta1=beta1,beta2=beta2,beta3=beta3,eps=eps,debiased=debiased,alpha=alpha,mode=mode)
        super().__init__(defaults)

        self.add_projected_keys("grad", "exp_avg")
        self.add_projected_keys("grad_sq", "exp_avg_sq")
        self.add_projected_keys("grad_cu", "exp_avg_cu")

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        beta1,beta2,beta3,eps,alpha=unpack_dicts(settings, 'beta1','beta2','beta3','eps','alpha', cls=NumberList)
        exp_avg, exp_avg_sq, exp_avg_cu = unpack_states(states, tensors, 'exp_avg', 'exp_avg_sq', 'exp_avg_cu', cls=TensorList)

        return cubic_adam_(
            tensors=TensorList(tensors),
            exp_avg_=exp_avg,
            exp_avg_sq_=exp_avg_sq,
            exp_avg_cu_=exp_avg_cu,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            eps=eps,
            debiased=settings[0]['debiased'],
            step=step,

            mode=settings[0]["mode"]
        )

class SubspaceCubicAdam(LREOptimizerBase):
    """Runs cubic Adam in low rank eigenbasis."""
    def __init__(self, beta1=0.9, beta2=0.95, beta3=0.95, eps=1e-8, mode: _cubic_adam_mode = 'signed_cbrt', cautious:bool=False, exact_reproject:bool=True):
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.eps = eps
        self.cautious = cautious
        self.mode: _cubic_adam_mode = mode
        self.exact_reproject = exact_reproject

    def step(self, g, L, Q, state):
        g = Q.T @ g

        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(g)
            state["exp_avg_sq"] = torch.zeros_like(g)
            state["exp_avg_cu"] = torch.zeros_like(g)
            state["current_step"] = 1

        dir = cubic_adam_(
            tensors = TensorList([g]),
            exp_avg_ = TensorList([state["exp_avg"]]),
            exp_avg_sq_ = TensorList([state["exp_avg_sq"]]),
            exp_avg_cu_ = TensorList([state["exp_avg_cu"]]),
            alpha = 1,
            beta1 = self.beta1,
            beta2 = self.beta2,
            beta3 = self.beta3,
            eps = self.eps,
            debiased = True,
            step = state["current_step"],

            mode=self.mode,
        )[0]

        state["current_step"] += 1
        return Q @ dir

    def reproject(self, L_old, Q_old, L_new, Q_new, state):
        if  "exp_avg" not in state: return

        C = Q_new.T @ Q_old

        state["exp_avg"] = C @ state["exp_avg"]
        state["exp_avg_sq"] = _squared_reproject(C, state["exp_avg_sq"], exact=self.exact_reproject)
        state["exp_avg_cu"] = C.pow(3) @ state["exp_avg_cu"] # exact reproject with 1_000_000 is feasible
