from operator import itemgetter
import torch

from ...core import Module, Target, Transform, apply_transform, Chainable
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states
from ..functional import (
    debias, debiased_step_size,
    ema_,
)



def adam_expclogmul_(
    tensors: TensorList,
    exp_avg_: TensorList,
    exp_avg_xpx_: TensorList,
    alpha: float | NumberList,
    beta1: float | NumberList,
    beta2: float | NumberList,
    eps: float | NumberList,
    step: int,
    debiased: bool = True,
    max_exp_avg_xpx_: TensorList | None = None,

    # inner args
    inner: Module | None = None,
    params: list[torch.Tensor] | None = None,
    grads: list[torch.Tensor] | None = None,
):
    """Returns new tensors."""
    tensors_abs = tensors.abs().add_(1e-4).clip(max=50)
    tensors_xpx = tensors_abs.log().square_().exp_()
    exp_avg_xpx_.lerp_(tensors_xpx, 1-beta2)

    if max_exp_avg_xpx_ is not None:
        max_exp_avg_xpx_.maximum_(exp_avg_xpx_)
        exp_avg_xpx_ = max_exp_avg_xpx_

    if inner is not None:
        assert params is not None
        tensors = TensorList(apply_transform(inner, tensors, params=params, grads=grads))

    exp_avg_ = ema_(tensors, exp_avg_=exp_avg_, beta=beta1, dampening=0,lerp=True)
    if debiased: alpha = debiased_step_size(step, beta1=beta1, beta2=beta2, alpha=alpha)

    exp_avg_xpx_ = exp_avg_xpx_.log().clip(min=0).sqrt_().exp()

    return (exp_avg_.lazy_mul(alpha) / exp_avg_xpx_.add_(eps))

class AdamExpclogmul(Transform):
    """Adam but uses exp(ln(|x|)^2) and exp(sqrt(ln(x))).

    Args:
        beta1 (float, optional): momentum. Defaults to 0.9.
        beta2 (float, optional): second momentum. Defaults to 0.999.
        eps (float, optional): epsilon. Defaults to 1e-8.
        alpha (float, optional): learning rate. Defaults to 1.
        amsgrad (bool, optional): Whether to divide by maximum of EMA of gradient squares instead. Defaults to False.
        debiased (bool, optional): whether to apply debiasing to momentums based on current step. Defaults to True.
    """
    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad: bool = False,
        alpha: float = 1.,
        debiased: bool = True,
        inner: Chainable | None = None
    ):
        defaults=dict(beta1=beta1,beta2=beta2,eps=eps,alpha=alpha,amsgrad=amsgrad,pow=pow,debiased=debiased)
        super().__init__(defaults, uses_grad=False)

        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        beta1,beta2,eps,alpha=unpack_dicts(settings, 'beta1','beta2','eps','alpha', cls=NumberList)
        amsgrad,debiased = itemgetter('amsgrad','debiased')(settings[0])

        if amsgrad:
            exp_avg, exp_avg_xpx, max_exp_avg_xpx = unpack_states(states, tensors, 'exp_avg', 'exp_avg_xpx', 'max_exp_avg_xpx', cls=TensorList)
        else:
            exp_avg, exp_avg_xpx = unpack_states(states, tensors, 'exp_avg', 'exp_avg_xpx', cls=TensorList)
            max_exp_avg_xpx = None


        return adam_expclogmul_(
            tensors=TensorList(tensors),
            exp_avg_=exp_avg,
            exp_avg_xpx_=exp_avg_xpx,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            step=step,
            debiased=debiased,
            max_exp_avg_xpx_=max_exp_avg_xpx,

            # inner args
            inner=self.children.get("inner", None),
            params=params,
            grads=grads,

        )
