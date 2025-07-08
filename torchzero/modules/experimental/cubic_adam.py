import torch

from ...core import Transform
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states


def signed_cbrt(x: TensorList) -> TensorList:
    return x.sign() * x.abs().pow(1/3)

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
    A = signed_cbrt(m3)
    B = m2.sqrt()
    C = m1
    discriminant = B.pow(2) - 4 * A * C

    denom = 2 * A
    root = discriminant.clamp(min=0).sqrt_()

    x0 = (-B + root) / (denom + eps)
    x1 = (-B - root) / (denom + eps)

    f0 = (A/3)*x0**3 + (B/2)*x0**2 + C*x0
    f1 = (A/3)*x1**3 + (B/2)*x1**2 + C*x1

    x_star = x0.where(f0 < f1, x1)

    adam = -C / (B + eps)
    x_star = adam.where(discriminant < 0, x_star)

    return x_star.mul_(-alpha)

class CubicAdam(Transform):
    """Adam which has 3rd momentum and minimizes a cubic polynomial.

    VERDICT: can outperform Adam very slightly. Usually very similar performance.

    .. warning::
        Experimental.

    """
    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.99,
        beta3: float = 0.99,
        eps: float = 1e-8,
        debiased:bool=True,
        alpha: float = 1.,
    ):
        defaults=dict(beta1=beta1,beta2=beta2,beta3=beta3,eps=eps,debiased=debiased,alpha=alpha)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
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
        )
