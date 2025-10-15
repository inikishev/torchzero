import torch

from ...core import TensorTransform, Chainable
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states

def adan_update_(
    g: TensorList,
    g_prev_: TensorList,
    m_: TensorList, # exponential moving average
    v_: TensorList, # exponential moving average of gradient differences
    n_: TensorList, # kinda like squared momentum
    beta1: float | NumberList,
    beta2: float | NumberList,
    beta3: float | NumberList,
    step: int,
):
    m_.lerp_(g, 1 - beta1)

    if step == 1:
        term = g
    else:
        diff = g - g_prev_
        v_.lerp_(diff, 1 - beta2)
        term = g + beta2 * diff

    n_.mul_(beta3).addcmul_(term, term, value=(1 - beta3))
    g_prev_.copy_(g)

def adan_apply_(
    m_: TensorList, # exponential moving average
    v_: TensorList, # exponential moving average of gradient differences
    n_: TensorList, # kinda like squared momentum
    beta1: float | NumberList,
    beta2: float | NumberList,
    beta3: float | NumberList,
    eps: float | NumberList,
    step: int,
):
    m = m_ / (1.0 - beta1**step)
    v = v_ / (1.0 - beta2**step)
    n = n_ / (1.0 - beta3**step)

    denom = n.sqrt_().add_(eps)
    num = m + beta2 * v

    update = num.div_(denom)

    return update



class Adan(TensorTransform):
    """Adaptive Nesterov Momentum Algorithm from https://arxiv.org/abs/2208.06677

    Args:
        beta1 (float, optional): momentum. Defaults to 0.98.
        beta2 (float, optional): momentum for gradient differences. Defaults to 0.92.
        beta3 (float, optional): thrid (squared) momentum. Defaults to 0.99.
        eps (float, optional): epsilon. Defaults to 1e-8.

    Example:
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Adan(),
        tz.m.LR(1e-3),
    )
    ```
    Reference:
        [Xie, X., Zhou, P., Li, H., Lin, Z., & Yan, S. (2024). Adan: Adaptive nesterov momentum algorithm for faster optimizing deep models. IEEE Transactions on Pattern Analysis and Machine Intelligence](https://arxiv.org/abs/2208.06677).
    """
    def __init__(
        self,
        beta1: float = 0.98,
        beta2: float = 0.92,
        beta3: float = 0.99,
        eps: float = 1e-8,

        m_tfm: Chainable | None = None,
        v_tfm: Chainable | None = None,
        n_tfm: Chainable | None = None,
    ):
        defaults=dict(beta1=beta1, beta2=beta2, beta3=beta3, eps=eps)
        super().__init__(defaults, uses_grad=False)

        self.set_child("m", m_tfm)
        self.set_child("v", v_tfm)
        self.set_child("n", n_tfm)

        self.add_projected_keys("grad_sq", "m", "v", "g_prev")
        self.add_projected_keys("grad", "n")

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        step = self.increment_counter("step", start=0)

        beta1, beta2, beta3 = unpack_dicts(settings, 'beta1','beta2','beta3', cls=NumberList)
        g_prev, m, v, n = unpack_states(states, tensors, 'g_prev', 'm', 'v', 'n', cls=TensorList)

        adan_update_(g=tensors, g_prev_=g_prev, m_=m, v_=v, n_=n, beta1=beta1, beta2=beta2, beta3=beta3, step=step+1)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        step = self.global_state["step"] # 0 on 1st step

        beta1, beta2, beta3, eps = unpack_dicts(settings, 'beta1','beta2','beta3', 'eps', cls=NumberList)
        m, v, n = unpack_states(states, tensors, 'm', 'v', 'n')

        # -------------------------------- transforms -------------------------------- #
        m = TensorList(self.inner_step_tensors("m", m, clone=True, params=params, grads=grads, loss=loss, must_exist=False))
        v = TensorList(self.inner_step_tensors("v", v, clone=True, params=params, grads=grads, loss=loss, must_exist=False))
        n = TensorList(self.inner_step_tensors("n", n, clone=True, params=params, grads=grads, loss=loss, must_exist=False))

        # ---------------------------------- update ---------------------------------- #
        return adan_apply_(m_=m, v_=v, n_=n, beta1=beta1, beta2=beta2, beta3=beta3, eps=eps, step=step+1)

