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
        eps=1e-6,
        tensorwise:bool=True,
        max_ema_growth: float | None = 1.5,
        ema_init: Literal['zeros', 'update'] = 'zeros',
        inner: Chainable | None = None,
    ):
        defaults = dict(beta=beta, ord=ord, tensorwise=tensorwise, ema_init=ema_init, eps=eps, max_ema_growth=max_ema_growth)
        super().__init__(defaults, inner=inner)

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        ord, tensorwise, ema_init, max_ema_growth = itemgetter('ord', 'tensorwise', 'ema_init', 'max_ema_growth')(settings[0])

        beta, eps = unpack_dicts(settings, 'beta', 'eps', cls=NumberList)

        ema = unpack_states(states, tensors, 'ema', init = (torch.zeros_like if ema_init=='zeros' else tensors), cls=TensorList)

        ema.lerp_(tensors, 1-beta)

        if tensorwise:
            ema_norm = ema.metric(ord)

            # clip ema norm growth
            if max_ema_growth is not None:
                prev_ema_norm = unpack_states(states, tensors, 'prev_ema_norm', init=ema_norm, cls=TensorList)
                allowed_norm = (prev_ema_norm * max_ema_growth).clip(min=1e-6)
                ema_denom = (ema_norm / allowed_norm).clip(min=1)
                ema.div_(ema_denom)
                ema_norm.div_(ema_denom)
                prev_ema_norm.set_(ema_norm)

            tensors_norm = tensors.norm(ord)
            denom = tensors_norm / ema_norm.clip(min=eps)
            if self.NORMALIZE: denom.clip_(min=eps)
            else: denom.clip_(min=1)

        else:
            ema_norm = ema.global_metric(ord)

            # clip ema norm growth
            if max_ema_growth is not None:
                prev_ema_norm = self.global_state.setdefault('prev_ema_norm', ema_norm)
                allowed_norm = prev_ema_norm * max_ema_growth
                if ema_norm > allowed_norm:
                    ema.div_(ema_norm / allowed_norm)
                    ema_norm = allowed_norm
                prev_ema_norm.set_(ema_norm)

            tensors_norm = tensors.global_metric(ord)
            denom = tensors_norm / ema_norm.clip(min=eps[0])
            if self.NORMALIZE: denom.clip_(min=eps[0])
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
        init: Literal['zeros', 'update'] = 'zeros',

        inner: Chainable | None = None,
        exp_avg_tfm:Chainable | None=None,
    ):
        defaults = dict(beta=beta, init=init)
        super().__init__(defaults, inner=inner)

        self.set_child('exp_avg', exp_avg_tfm)

    def single_tensor_initialize(self, tensor, param, grad, loss, state, setting):
        if setting["init"] == "zeros":
            state["exp_avg"] = torch.zeros_like(tensor)
        else:
            state["exp_avg"] = tensor.abs()

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
