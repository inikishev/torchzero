from collections import deque
from operator import itemgetter
from typing import Literal

import torch

from ...core import Target, Transform
from ...utils import TensorList, NumberList, unpack_dicts, unpack_states
from ..functional import debias, ema_, ema_sq_, sqrt_ema_sq_, centered_ema_sq_, sqrt_centered_ema_sq_, debias_second_momentum


class EMA(Transform):
    """Maintains an exponential moving average of update.

    Args:
        momentum (float, optional): momentum (beta). Defaults to 0.9.
        dampening (float, optional): momentum dampening. Defaults to 0.
        debiased (bool, optional): whether to debias the EMA like in Adam. Defaults to False.
        lerp (bool, optional): whether to use linear interpolation. Defaults to True.
        ema_init (str, optional): initial values for the EMA, "zeros" or "update".
        target (Target, optional): target to apply EMA to. Defaults to 'update'.
    """
    def __init__(self, momentum:float=0.9, dampening:float=0, debiased: bool = False, lerp=True, ema_init: Literal['zeros', 'update'] = 'zeros', target: Target = 'update'):
        defaults = dict(momentum=momentum,dampening=dampening,debiased=debiased,lerp=lerp,ema_init=ema_init)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        debiased, lerp, ema_init = itemgetter('debiased','lerp','ema_init')(settings[0])

        exp_avg = unpack_states(states, tensors, 'exp_avg',
                                init=torch.zeros_like if ema_init=='zeros' else tensors, cls=TensorList)
        momentum, dampening = unpack_dicts(settings, 'momentum','dampening', cls=NumberList)

        exp_avg = ema_(TensorList(tensors), exp_avg_=exp_avg,beta=momentum,dampening=dampening,lerp=lerp)

        if debiased: return debias(exp_avg, step=step, beta1=momentum, alpha=1, inplace=False)
        else: return exp_avg.clone() # this has exp_avg storage so needs to be cloned


class EMASquared(Transform):
    """Maintains an exponential moving average of squared updates.

    Args:
        beta (float, optional): momentum value. Defaults to 0.999.
        amsgrad (bool, optional): whether to maintain maximum of the exponential moving average. Defaults to False.
        pow (float, optional): power, absolute value is always used. Defaults to 2.
    """
    EMA_SQ_FN: staticmethod = staticmethod(ema_sq_)

    def __init__(self, beta:float=0.999, amsgrad=False, pow:float=2):
        defaults = dict(beta=beta,pow=pow,amsgrad=amsgrad)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        amsgrad, pow = itemgetter('amsgrad', 'pow')(self.settings[params[0]])
        beta = NumberList(s['beta'] for s in settings)

        if amsgrad:
            exp_avg_sq, max_exp_avg_sq = unpack_states(states, tensors, 'exp_avg_sq', 'max_exp_avg_sq', cls=TensorList)
        else:
            exp_avg_sq = unpack_states(states, tensors, 'exp_avg_sq', cls=TensorList)
            max_exp_avg_sq = None

        return self.EMA_SQ_FN(TensorList(tensors), exp_avg_sq_=exp_avg_sq, beta=beta, max_exp_avg_sq_=max_exp_avg_sq, pow=pow).clone()

class SqrtEMASquared(Transform):
    """Maintains an exponential moving average of squared updates, outputs optionally debiased square root.

    Args:
        beta (float, optional): momentum value. Defaults to 0.999.
        amsgrad (bool, optional): whether to maintain maximum of the exponential moving average. Defaults to False.
        debiased (bool, optional): whether to multiply the output by a debiasing term from the Adam method. Defaults to False.
        pow (float, optional): power, absolute value is always used. Defaults to 2.
    """
    SQRT_EMA_SQ_FN: staticmethod = staticmethod(sqrt_ema_sq_)
    def __init__(self, beta:float=0.999, amsgrad=False, debiased: bool = False, pow:float=2,):
        defaults = dict(beta=beta,pow=pow,amsgrad=amsgrad,debiased=debiased)
        super().__init__(defaults, uses_grad=False)


    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        amsgrad, pow, debiased = itemgetter('amsgrad', 'pow', 'debiased')(settings[0])
        beta = NumberList(s['beta'] for s in settings)

        if amsgrad:
            exp_avg_sq, max_exp_avg_sq = unpack_states(states, tensors, 'exp_avg_sq', 'max_exp_avg_sq', cls=TensorList)
        else:
            exp_avg_sq = unpack_states(states, tensors, 'exp_avg_sq', cls=TensorList)
            max_exp_avg_sq = None

        return self.SQRT_EMA_SQ_FN(
            TensorList(tensors),
            exp_avg_sq_=exp_avg_sq,
            beta=beta,
            max_exp_avg_sq_=max_exp_avg_sq,
            debiased=debiased,
            step=step,
            pow=pow,
        )


class Debias(Transform):
    """Multiplies the update by an Adam debiasing term based first and/or second momentum.

    Args:
        beta1 (float | None, optional):
            first momentum, should be the same as first momentum used in modules before. Defaults to None.
        beta2 (float | None, optional):
            second (squared) momentum, should be the same as second momentum used in modules before. Defaults to None.
        alpha (float, optional): learning rate. Defaults to 1.
        pow (float, optional): power, assumes absolute value is used. Defaults to 2.
        target (Target, optional): target. Defaults to 'update'.
    """
    def __init__(self, beta1: float | None = None, beta2: float | None = None, alpha: float = 1, pow:float=2, target: Target = 'update',):
        defaults = dict(beta1=beta1, beta2=beta2, alpha=alpha, pow=pow)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        pow = settings[0]['pow']
        alpha, beta1, beta2 = unpack_dicts(settings, 'alpha', 'beta1', 'beta2', cls=NumberList)

        return debias(TensorList(tensors), step=step, beta1=beta1, beta2=beta2, alpha=alpha, pow=pow, inplace=True)

class Debias2(Transform):
    """Multiplies the update by an Adam debiasing term based on the second momentum.

    Args:
        beta (float | None, optional):
            second (squared) momentum, should be the same as second momentum used in modules before. Defaults to None.
        pow (float, optional): power, assumes absolute value is used. Defaults to 2.
        target (Target, optional): target. Defaults to 'update'.
    """
    def __init__(self, beta: float = 0.999, pow: float = 2, target: Target = 'update',):
        defaults = dict(beta=beta, pow=pow)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        pow = settings[0]['pow']
        beta = NumberList(s['beta'] for s in settings)
        return debias_second_momentum(TensorList(tensors), step=step, beta=beta, pow=pow, inplace=True)

class CenteredEMASquared(Transform):
    """Maintains a centered exponential moving average of squared updates. This also maintains an additional
    exponential moving average of un-squared updates, square of which is subtracted from the EMA.

    Args:
        beta (float, optional): momentum value. Defaults to 0.999.
        amsgrad (bool, optional): whether to maintain maximum of the exponential moving average. Defaults to False.
        pow (float, optional): power, absolute value is always used. Defaults to 2.
    """
    def __init__(self, beta: float = 0.99, amsgrad=False, pow:float=2):
        defaults = dict(beta=beta, amsgrad=amsgrad, pow=pow)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        amsgrad, pow = itemgetter('amsgrad', 'pow')(settings[0])
        beta = NumberList(s['beta'] for s in settings)

        if amsgrad:
            exp_avg, exp_avg_sq, max_exp_avg_sq = unpack_states(states, tensors, 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq', cls=TensorList)
        else:
            exp_avg, exp_avg_sq = unpack_states(states, tensors, 'exp_avg', 'exp_avg_sq', cls=TensorList)
            max_exp_avg_sq = None

        return centered_ema_sq_(
            TensorList(tensors),
            exp_avg_=exp_avg,
            exp_avg_sq_=exp_avg_sq,
            beta=beta,
            max_exp_avg_sq_=max_exp_avg_sq,
            pow=pow,
        ).clone()

class CenteredSqrtEMASquared(Transform):
    """Maintains a centered exponential moving average of squared updates, outputs optionally debiased square root.
    This also maintains an additional exponential moving average of un-squared updates, square of which is subtracted from the EMA.

    Args:
        beta (float, optional): momentum value. Defaults to 0.999.
        amsgrad (bool, optional): whether to maintain maximum of the exponential moving average. Defaults to False.
        debiased (bool, optional): whether to multiply the output by a debiasing term from the Adam method. Defaults to False.
        pow (float, optional): power, absolute value is always used. Defaults to 2.
    """
    def __init__(self, beta: float = 0.99, amsgrad=False, debiased: bool = False, pow:float=2):
        defaults = dict(beta=beta, amsgrad=amsgrad, debiased=debiased, pow=pow)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        amsgrad, pow, debiased = itemgetter('amsgrad', 'pow', 'debiased')(settings[0])
        beta = NumberList(s['beta'] for s in settings)

        if amsgrad:
            exp_avg, exp_avg_sq, max_exp_avg_sq = unpack_states(states, tensors, 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq', cls=TensorList)
        else:
            exp_avg, exp_avg_sq = unpack_states(states, tensors, 'exp_avg', 'exp_avg_sq', cls=TensorList)
            max_exp_avg_sq = None

        return sqrt_centered_ema_sq_(
            TensorList(tensors),
            exp_avg_=exp_avg,
            exp_avg_sq_=exp_avg_sq,
            beta=beta,
            debiased=debiased,
            step=step,
            max_exp_avg_sq_=max_exp_avg_sq,
            pow=pow,
        )