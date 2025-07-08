import torch

from ...core import Transform
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states

def adan_(
    g: TensorList,
    g_prev_: TensorList,
    m_: TensorList, # exponential moving average
    v_: TensorList, # exponential moving average of gradient differences
    n_: TensorList, # kinda like squared momentum
    n_prev_: TensorList | None,
    beta1: float | NumberList,
    beta2: float | NumberList,
    beta3: float | NumberList,
    eps: float | NumberList,
    use_n_prev: bool,
):
    """Returns new tensors."""
    m_.lerp_(g, 1-beta1)

    y = g - g_prev_
    v_.lerp_(y, 1-beta2)

    y.mul_(1-beta2).add_(g)
    n_.mul_(beta3).addcmul_(y, y, 1-beta3)

    if use_n_prev:
        assert n_prev_ is not None
        ns = n_prev_.clone()
        n_prev_.copy_(n_)
        n_ = ns

    eta = n_.sqrt().add_(eps).reciprocal_()
    term = m_ + (1-beta2)*v_
    update = eta.mul_(term)

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

    Reference:
        Xie, X., Zhou, P., Li, H., Lin, Z., & Yan, S. (2024). Adan: Adaptive nesterov momentum algorithm for faster optimizing deep models. IEEE Transactions on Pattern Analysis and Machine Intelligence. https://arxiv.org/abs/2208.06677
    """
    def __init__(
        self,
        beta1: float = 0.98,
        beta2: float = 0.92,
        beta3: float = 0.99,
        eps: float = 1e-8,
        use_n_prev: bool = False,
    ):
        defaults=dict(beta1=beta1,beta2=beta2,beta3=beta3,eps=eps,use_n_prev=use_n_prev)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        beta1,beta2,beta3,eps=unpack_dicts(settings, 'beta1','beta2','beta3','eps', cls=NumberList)
        s = settings[0]
        use_n_prev = s['use_n_prev']

        g_prev, m, v, n = unpack_states(states, tensors, 'g_prev','m','v','n', cls=TensorList)


        if use_n_prev:
            n_prev = unpack_states(states, tensors, 'n_prev', cls=TensorList)
        else:
            n_prev = None

        if step == 1:
            # initial values, also runs on restarts
            m.copy_(tensors)
            n.set_(tensors ** 2)
            v.zero_()
            g_prev.copy_(tensors)
            if n_prev is not None: n_prev.set_(tensors ** 2)

        if step == 2:
            v.set_(tensors - g_prev)

        update = adan_(
            g=tensors,
            g_prev_=g_prev,
            m_=m,
            v_=v,
            n_=n,
            n_prev_=n_prev,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            eps=eps,
            use_n_prev=use_n_prev,
        )

        return update