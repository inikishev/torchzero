import torch

from ...core import Transform
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states

def adan_(
    g: TensorList,
    g_prev_: TensorList,
    m_: TensorList, # exponential moving average
    v_: TensorList, # exponential moving average of gradient differences
    n_: TensorList, # kinda like squared momentum
    beta1: float | NumberList,
    beta2: float | NumberList,
    beta3: float | NumberList,
    eps: float | NumberList,
    step: int,
):
    """Returns new tensors"""
    m_.lerp_(g, 1 - beta1)

    if step == 1:
        term = g
    else:
        diff = g - g_prev_
        v_.lerp_(diff, 1 - beta2)
        term = g + beta2 * diff

    n_.mul_(beta3).addcmul_(term, term, value=(1 - beta3))

    m = m_ / (1.0 - beta1**step)
    v = v_ / (1.0 - beta2**step)
    n = n_ / (1.0 - beta3**step)

    denom = n.sqrt_().add_(eps)
    num = m + beta2 * v

    update = num.div_(denom)
    g_prev_.copy_(g)

    return update



class Adan(Transform):
    """Adaptive Nesterov Momentum Algorithm from https://arxiv.org/abs/2208.06677

    Args:
        beta1 (float, optional): momentum. Defaults to 0.98.
        beta2 (float, optional): momentum for gradient differences. Defaults to 0.92.
        beta3 (float, optional): thrid (squared) momentum. Defaults to 0.99.
        eps (float, optional): epsilon. Defaults to 1e-8.
        use_n_prev (bool, optional):
            whether to use previous gradient differences momentum.

    Example:
    ```python
    opt = tz.Modular(
        model.parameters(),
        tz.m.Adan(),
        tz.m.LR(1e-3),
    )
    Reference:
        Xie, X., Zhou, P., Li, H., Lin, Z., & Yan, S. (2024). Adan: Adaptive nesterov momentum algorithm for faster optimizing deep models. IEEE Transactions on Pattern Analysis and Machine Intelligence. https://arxiv.org/abs/2208.06677
    """
    def __init__(
        self,
        beta1: float = 0.98,
        beta2: float = 0.92,
        beta3: float = 0.99,
        eps: float = 1e-8,
    ):
        defaults=dict(beta1=beta1,beta2=beta2,beta3=beta3,eps=eps)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        beta1,beta2,beta3,eps=unpack_dicts(settings, 'beta1','beta2','beta3','eps', cls=NumberList)
        g_prev, m, v, n = unpack_states(states, tensors, 'g_prev','m','v','n', cls=TensorList)

        update = adan_(
            g=tensors,
            g_prev_=g_prev,
            m_=m,
            v_=v,
            n_=n,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            eps=eps,
            step=step,
        )

        return update