from collections import deque
from operator import itemgetter
from typing import Literal

import torch

from ...core import Target, Transform
from ...utils import TensorList, NumberList
from ..functional import debias1, debias2, ema_, ema_sq_, sqrt_ema_sq_, centered_ema_sq_, sqrt_centered_ema_sq_


class EMA(Transform):
    """Maintains EMA of update.

    Args:
        momentum (float, optional): momentum (beta). Defaults to 0.9.
        dampening (float, optional): momentum dampening. Defaults to 0.
        debiased (bool, optional): whether to debias the EMA like in Adam. Defaults to False.
        lerp (bool, optional): whether to use linear interpolation. Defaults to True.
        target (Target, optional): target to apply EMA to. Defaults to 'update'.
    """
    def __init__(self, momentum:float=0.9, dampening:float=0, debiased: bool = False, lerp=True, ema_init: Literal['zeros', 'update'] = 'zeros', target: Target = 'update'):
        defaults = dict(momentum=momentum,dampening=dampening,debiased=debiased,lerp=lerp,ema_init=ema_init)
        super().__init__(defaults, uses_grad=False, target=target)
        self.current_step = 0

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        self.current_step += 1

        debiased, lerp, ema_init = itemgetter('debiased','lerp','ema_init')(self.settings[params[0]])

        exp_avg = self.get_state('exp_avg', params=params, init=torch.zeros_like if ema_init=='zeros' else target, cls=TensorList)
        momentum, dampening = self.get_settings('momentum','dampening', params=params, cls=NumberList)

        exp_avg = ema_(TensorList(target), exp_avg_=exp_avg,beta=momentum,dampening=dampening,lerp=lerp)

        if debiased: return debias1(exp_avg, step=self.current_step, beta=momentum, alpha=1, inplace=False)
        else: return exp_avg.clone() # this has exp_avg storage so needs to be cloned


class EMASquared(Transform):
    EMA_SQ_FN: staticmethod = staticmethod(ema_sq_)

    def __init__(self, beta:float=0.999, amsgrad=False, pow:float=2, target: Target = 'update'):
        defaults = dict(beta=beta,pow=pow,amsgrad=amsgrad)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        amsgrad, pow = itemgetter('amsgrad', 'pow')(self.settings[params[0]])
        beta = self.get_settings('beta', params=params, cls=NumberList)

        if amsgrad:
            exp_avg_sq, max_exp_avg_sq = self.get_state('exp_avg_sq', 'max_exp_avg_sq', params=params, cls=TensorList)
        else:
            exp_avg_sq = self.get_state('exp_avg_sq', params=params, cls=TensorList)
            max_exp_avg_sq = None

        return self.EMA_SQ_FN(TensorList(target), exp_avg_sq_=exp_avg_sq, beta=beta, max_exp_avg_sq_=max_exp_avg_sq, pow=pow).clone()

class SqrtEMASquared(Transform):
    SQRT_EMA_SQ_FN: staticmethod = staticmethod(sqrt_ema_sq_)

    def __init__(self, beta:float=0.999, amsgrad=False, debiased: bool = False, pow:float=2, target: Target = 'update',):
        defaults = dict(beta=beta,pow=pow,amsgrad=amsgrad,debiased=debiased)
        super().__init__(defaults, uses_grad=False, target=target)
        self.global_state['current_step'] = 0

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        self.global_state['current_step'] += 1
        amsgrad, pow, debiased = itemgetter('amsgrad', 'pow', 'debiased')(self.settings[params[0]])
        beta = self.get_settings('beta', params=params, cls=NumberList)

        if amsgrad:
            exp_avg_sq, max_exp_avg_sq = self.get_state('exp_avg_sq', 'max_exp_avg_sq', params=params, cls=TensorList)
        else:
            exp_avg_sq = self.get_state('exp_avg_sq', params=params, cls=TensorList)
            max_exp_avg_sq = None

        return self.SQRT_EMA_SQ_FN(TensorList(target),exp_avg_sq_=exp_avg_sq,beta=beta,max_exp_avg_sq_=max_exp_avg_sq,debiased=debiased,step=self.global_state['current_step'],pow=pow)

class Debias1(Transform):
    def __init__(self, beta: float = 0.9, alpha: float = 1, target: Target = 'update',):
        defaults = dict(beta=beta, alpha=alpha)
        super().__init__(defaults, uses_grad=False, target=target)
        self.global_state['step'] = 0

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        self.global_state['step'] = self.global_state.setdefault('step', 0) + 1

        alpha = self.settings[params[0]]['alpha']
        beta = self.get_settings('beta', params=params, cls=NumberList)
        return debias1(TensorList(target), step=self.global_state['step'], beta=beta, alpha=alpha, inplace=True)

class Debias2(Transform):
    def __init__(self, beta: float = 0.999, pow: float = 2, target: Target = 'update',):
        defaults = dict(beta=beta, pow=pow)
        super().__init__(defaults, uses_grad=False, target=target)
        self.global_state['step'] = 0

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        self.global_state['step'] = self.global_state.setdefault('step', 0) + 1

        pow = self.settings[params[0]]['pow']
        beta = self.get_settings('beta', params=params, cls=NumberList)
        return debias2(TensorList(target), step=self.global_state['step'], beta=beta, pow=pow, inplace=True)


class CenteredEMASquared(Transform):
    def __init__(self, beta: float = 0.99, amsgrad=False, pow:float=2, target: Target = 'update'):
        defaults = dict(beta=beta, amsgrad=amsgrad, pow=pow)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        amsgrad, pow = itemgetter('amsgrad', 'pow')(self.settings[params[0]])
        beta = self.get_settings('beta', params=params, cls=NumberList)

        if amsgrad:
            exp_avg, exp_avg_sq, max_exp_avg_sq = self.get_state('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq', params=params, cls=TensorList)
        else:
            exp_avg, exp_avg_sq = self.get_state('exp_avg', 'exp_avg_sq', params=params, cls=TensorList)
            max_exp_avg_sq = None

        return centered_ema_sq_(
            TensorList(target),
            exp_avg_=exp_avg,
            exp_avg_sq_=exp_avg_sq,
            beta=beta,
            max_exp_avg_sq_=max_exp_avg_sq,
            pow=pow,
        ).clone()

class CenteredSqrtEMASquared(Transform):
    def __init__(self, beta: float = 0.99, amsgrad=False, debiased: bool = False, pow:float=2, target: Target = 'update'):
        defaults = dict(beta=beta, amsgrad=amsgrad, debiased=debiased, pow=pow)
        super().__init__(defaults, uses_grad=False, target=target)
        self.global_state['step'] = 0

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        self.global_state['step'] = self.global_state.setdefault('step', 0) + 1
        amsgrad, pow, debiased = itemgetter('amsgrad', 'pow', 'debiased')(self.settings[params[0]])
        beta = self.get_settings('beta', params=params, cls=NumberList)

        if amsgrad:
            exp_avg, exp_avg_sq, max_exp_avg_sq = self.get_state('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq', params=params, cls=TensorList)
        else:
            exp_avg, exp_avg_sq = self.get_state('exp_avg', 'exp_avg_sq', params=params, cls=TensorList)
            max_exp_avg_sq = None

        return sqrt_centered_ema_sq_(
            TensorList(target),
            exp_avg_=exp_avg,
            exp_avg_sq_=exp_avg_sq,
            beta=beta,
            debiased=debiased,
            step=self.global_state['step'],
            max_exp_avg_sq_=max_exp_avg_sq,
            pow=pow,
        )