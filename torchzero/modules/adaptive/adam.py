import torch

from ...core import Chainable, Module, TensorTransform
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states
from ..opt_utils import debiased_step_size


class Adam(TensorTransform):
    """Adam. Divides gradient EMA by EMA of gradient squares with debiased step size.

    This implementation is identical to :code:`torch.optim.Adam`.

    Args:
        beta1 (float, optional): momentum. Defaults to 0.9.
        beta2 (float, optional): second momentum. Defaults to 0.999.
        eps (float, optional): epsilon. Defaults to 1e-8.
        alpha (float, optional): learning rate. Defaults to 1.
        amsgrad (bool, optional): Whether to divide by maximum of EMA of gradient squares instead. Defaults to False.
        pow (float, optional): power used in second momentum power and root. Defaults to 2.
        debias (bool, optional): whether to apply debiasing to momentums based on current step. Defaults to True.
    """
    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad: bool = False,
        alpha: float = 1.,
        debias: bool = True,

        exp_avg_tfm: Chainable | None = None,
        exp_avg_sq_tfm: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults["exp_avg_tfm"], defaults["exp_avg_sq_tfm"]
        super().__init__(defaults)

        self.set_child('exp_avg', exp_avg_tfm)
        self.set_child('exp_avg_sq', exp_avg_sq_tfm)

        self.add_projected_keys("grad", "exp_avg")
        self.add_projected_keys("grad_sq", "exp_avg_sq", "max_exp_avg_sq")

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        self.increment_counter("step", start=0)
        beta1, beta2 = unpack_dicts(settings, 'beta1','beta2', cls=NumberList)

        # ----------------------------- initialize states ---------------------------- #
        if settings[0]["amsgrad"]:
            exp_avg, exp_avg_sq, max_exp_avg_sq = unpack_states(
                states, tensors, 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq', cls=TensorList)
        else:
            exp_avg, exp_avg_sq = unpack_states(states, tensors, 'exp_avg', 'exp_avg_sq', cls=TensorList)
            max_exp_avg_sq = None

        # ------------------------------ update moments ------------------------------ #
        exp_avg.lerp_(tensors, weight=1-beta1)
        exp_avg_sq.mul_(beta2).addcmul_(tensors, tensors, value=1-beta2)

        if max_exp_avg_sq is not None:
            max_exp_avg_sq.maximum_(exp_avg_sq)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        step = self.global_state["step"] # 0 on 1st step
        fs = settings[0]

        if fs["amsgrad"]: key = "max_exp_avg_sq"
        else: key = "exp_avg_sq"
        exp_avg, exp_avg_sq = unpack_states(states, tensors, 'exp_avg', key, cls=TensorList)
        beta1, beta2, alpha, eps = unpack_dicts(settings, 'beta1', 'beta2', 'alpha', 'eps', cls=NumberList)

        # -------------------------------- transforms -------------------------------- #
        exp_avg = TensorList(self.inner_step_tensors(
            "exp_avg", tensors=exp_avg, clone=True, params=params, grads=grads, loss=loss, must_exist=False))

        exp_avg_sq = TensorList(self.inner_step_tensors(
            "exp_avg_sq", tensors=exp_avg_sq, clone=True, params=params, grads=grads, loss=loss, must_exist=False))

        # ---------------------------------- debias ---------------------------------- #
        if fs["debias"]:
            alpha = debiased_step_size((step + 1), beta1=beta1, beta2=beta2, alpha=alpha)
            exp_avg = exp_avg * alpha

        # ---------------------------------- update ---------------------------------- #
        return exp_avg / exp_avg_sq.sqrt().add_(eps)

