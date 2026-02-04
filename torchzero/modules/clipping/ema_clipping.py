from collections.abc import Iterable, Sequence
from operator import itemgetter
from typing import Literal

import torch

from ...core import Chainable, TensorTransform, step
from ...utils import Metrics, NumberList, TensorList, unpack_dicts, unpack_states


class ClipNormByEMA(TensorTransform):
    """Clips norm to be no larger than the norm of an exponential moving average of past updates.

    Args:
        beta (float, optional): beta for the exponential moving average. Defaults to 0.99.
        ord (float, optional): order of the norm. Defaults to 2.
        eps (float, optional): epsilon for division. Defaults to 1e-6.
        tensorwise (bool, optional):
            if True, norms are calculated parameter-wise, otherwise treats all parameters as single vector. Defaults to True.
        max_ema_growth (float | None, optional):
            if specified, restricts how quickly exponential moving average norm can grow. The norm is allowed to grow by at most this value per step. Defaults to 1.5.
        ema_init (str, optional):
            How to initialize exponential moving average on first step, "update" to use the first update or "zeros". Defaults to 'zeros'.
    """
    NORMALIZE = False
    def __init__(
        self,
        beta=0.99,
        ord: Metrics = 2,
        tensorwise:bool=True,
        max_ema_growth: float | None = 1.5,
        init: float = 0.0,
        min_norm: float = 1e-6,

        inner: Chainable | None = None,
    ):
        defaults = dict(beta=beta, ord=ord, tensorwise=tensorwise, init=init, min_norm=min_norm, max_ema_growth=max_ema_growth)
        super().__init__(defaults, inner=inner)
        self.add_projected_keys("grad", "exp_avg")

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        eps = torch.finfo(tensors[0].dtype).tiny * 2
        ord, tensorwise, init, max_ema_growth = itemgetter('ord', 'tensorwise', 'init', 'max_ema_growth')(settings[0])

        beta, min_norm = unpack_dicts(settings, 'beta', 'min_norm', cls=NumberList)

        exp_avg = unpack_states(states, tensors, 'exp_avg', init = lambda x: torch.full_like(x, init), cls=TensorList)

        exp_avg.lerp_(tensors, 1-beta)

        # ----------------------------- tensorwise update ---------------------------- #
        if tensorwise:
            tensors_norm = tensors.norm(ord)
            ema_norm = exp_avg.metric(ord)

            # clip ema norm growth
            if max_ema_growth is not None:
                prev_ema_norm = unpack_states(states, tensors, 'prev_ema_norm', init=ema_norm, cls=TensorList)
                allowed_norm = (prev_ema_norm * max_ema_growth).clip(min=min_norm)

                ema_denom = (ema_norm / allowed_norm).clip(min=1)
                exp_avg.div_(ema_denom)
                ema_norm.div_(ema_denom)

                prev_ema_norm.set_(ema_norm)


        # ------------------------------- global update ------------------------------ #
        else:
            tensors_norm = tensors.global_metric(ord)
            ema_norm = exp_avg.global_metric(ord)

            # clip ema norm growth
            if max_ema_growth is not None:
                prev_ema_norm = self.global_state.setdefault('prev_ema_norm', ema_norm)
                allowed_norm = (prev_ema_norm * max_ema_growth).clip(min=min_norm[0])

                if ema_norm > allowed_norm:
                    exp_avg.div_(ema_norm / allowed_norm)
                    ema_norm = allowed_norm

                prev_ema_norm.set_(ema_norm)


        # ------------------- compute denominator to clip/normalize ------------------ #
        denom = tensors_norm / ema_norm.clip(min=eps)
        if self.NORMALIZE: denom.clip_(min=eps)
        else: denom.clip_(min=1)
        self.global_state['denom'] = denom

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        denom = self.global_state.pop('denom')
        torch._foreach_div_(tensors, denom)
        return tensors

class NormalizeByEMA(ClipNormByEMA):
    """Sets norm of the update to be the same as the norm of an exponential moving average of past updates.

    Args:
        beta (float, optional): beta for the exponential moving average. Defaults to 0.99.
        ord (float, optional): order of the norm. Defaults to 2.
        eps (float, optional): epsilon for division. Defaults to 1e-6.
        tensorwise (bool, optional):
            if True, norms are calculated parameter-wise, otherwise treats all parameters as single vector. Defaults to True.
        max_ema_growth (float | None, optional):
            if specified, restricts how quickly exponential moving average norm can grow. The norm is allowed to grow by at most this value per step. Defaults to 1.5.
        ema_init (str, optional):
            How to initialize exponential moving average on first step, "update" to use the first update or "zeros". Defaults to 'zeros'.
    """
    NORMALIZE = True

# TODO Centralize by EMA?

class ClipValueByEMA(TensorTransform):
    """Clips magnitude of update to be no larger than magnitude of exponential moving average of past (unclipped) updates.

    Args:
        beta (float, optional): beta for the exponential moving average. Defaults to 0.99.
        ema_init (str, optional):
            How to initialize exponential moving average on first step,
            "update" to use the first update or "zeros". Defaults to 'zeros'.
        exp_avg_tfm (Chainable | None, optional):
            optional modules applied to exponential moving average before clipping by it. Defaults to None.
    """
    def __init__(
        self,
        beta=0.99,
        init: float = 0,

        inner: Chainable | None = None,
        exp_avg_tfm:Chainable | None=None,
    ):
        defaults = dict(beta=beta, init=init)
        super().__init__(defaults, inner=inner)

        self.set_child('exp_avg', exp_avg_tfm)
        self.add_projected_keys("grad", "exp_avg")

    def single_tensor_initialize(self, tensor, param, grad, loss, state, setting):
        state["exp_avg"] = tensor.abs() * setting["init"]

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        beta = unpack_dicts(settings, 'beta', cls=NumberList)

        exp_avg = unpack_states(states, tensors, 'exp_avg', must_exist=True, cls=TensorList)
        exp_avg.lerp_(tensors.abs(), 1-beta)

    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        exp_avg = unpack_states(states, tensors, 'exp_avg')

        exp_avg = TensorList(
            self.inner_step_tensors("exp_avg", exp_avg, clone=True, params=params, grads=grads, loss=loss, must_exist=False))

        tensors.clip_(-exp_avg, exp_avg)
        return tensors