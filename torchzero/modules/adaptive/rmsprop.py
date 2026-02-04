from typing import Literal

import torch

from ...core import TensorTransform, Chainable
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states

class RMSprop(TensorTransform):
    """Divides graient by EMA of gradient squares.

    This implementation is identical to :code:`torch.optim.RMSprop`.

    Args:
        smoothing (float, optional): beta for exponential moving average of gradient squares. Defaults to 0.99.
        eps (float, optional): epsilon for division. Defaults to 1e-8.
        centered (bool, optional): whether to center EMA of gradient squares using an additional EMA. Defaults to False.
        debias (bool, optional): applies Adam debiasing. Defaults to False.
        amsgrad (bool, optional): Whether to divide by maximum of EMA of gradient squares instead. Defaults to False.
        pow (float, optional): power used in second momentum power and root. Defaults to 2.
        init (str, optional): how to initialize EMA, either "update" to use first update or "zeros". Defaults to "update".
        inner (Chainable | None, optional):
            Inner modules that are applied after updating EMA and before preconditioning. Defaults to None.
    """
    def __init__(
        self,
        smoothing: float = 0.99,
        eps: float = 1e-8,
        centered: bool = False,
        debias: bool = False,
        amsgrad: bool = False,
        init: Literal["zeros", "update"] = "zeros",

        inner: Chainable | None = None,
        exp_avg_sq_tfm: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults["inner"], defaults["exp_avg_sq_tfm"]
        super().__init__(defaults, inner=inner)

        self.set_child('exp_avg_sq', exp_avg_sq_tfm)
        self.add_projected_keys("grad", "exp_avg")
        self.add_projected_keys("grad_sq", "exp_avg_sq", "exp_avg_sq_max")

    @torch.no_grad
    def single_tensor_initialize(self, tensor, param, grad, loss, state, setting):
        if setting["init"] == "zeros":
            state["exp_avg_sq"] = torch.zeros_like(tensor)
            if setting["centered"]: state["exp_avg"] = torch.zeros_like(tensor)
            if setting["amsgrad"]: state["amsgrad"] = torch.zeros_like(tensor)

        else:
            state["exp_avg_sq"] = tensor ** 2
            if setting["centered"]: state["exp_avg"] = tensor.clone()
            if setting["amsgrad"]: state["amsgrad"] = tensor ** 2

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        self.increment_counter("step", start = 0)
        fs = settings[0]

        exp_avg_sq = unpack_states(states, tensors, "exp_avg_sq", cls=TensorList)

        # update exponential average
        smoothing = NumberList(s["smoothing"] for s in settings)
        exp_avg_sq.mul_(smoothing).addcmul_(tensors, tensors, value=1-smoothing)

        # update mean estimate if centered
        if fs["centered"]:
            exp_avg = unpack_states(states, tensors, "exp_avg", cls=TensorList)
            exp_avg.lerp_(tensors, 1-smoothing)

        # amsgrad
        if fs["amsgrad"]:
            exp_avg_sq_max = unpack_states(states, tensors, "exp_avg_sq_max", cls=TensorList)
            exp_avg_sq_max.maximum_(exp_avg_sq)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        step = self.global_state["step"] # 0 on 1st step
        eps = NumberList(s["eps"] for s in settings)
        fs = settings[0]

        if fs["amsgrad"]: key = "max_exp_avg_sq"
        else: key = "exp_avg_sq"
        exp_avg_sq = TensorList(s[key] for s in states)

        # load mean estimate if centered
        exp_avg = None
        if fs['centered']:
            exp_avg = TensorList(s["exp_avg"] for s in states)

        # debias exp_avg_sq and exp_avg
        if fs["debias"]:
            smoothing = NumberList(s["smoothing"] for s in settings)
            bias_correction = 1 - (smoothing ** (step + 1))
            exp_avg_sq = exp_avg_sq / bias_correction

            if fs['centered']:
                assert exp_avg is not None
                exp_avg = exp_avg / bias_correction

        # apply transform to potentially debiased exp_avg_sq
        exp_avg_sq = TensorList(self.inner_step_tensors(
            "exp_avg_sq", exp_avg_sq, params=params, grads=grads, loss=loss, clone=True, must_exist=False
        ))

        # center
        if fs["centered"]:
            assert exp_avg is not None
            exp_avg_sq = exp_avg_sq.addcmul(exp_avg, exp_avg, value=-1)

        return tensors.div_(exp_avg_sq.sqrt().add_(eps))