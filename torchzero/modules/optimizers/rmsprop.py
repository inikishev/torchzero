from operator import itemgetter

from ...core import Module, Target, Transform
from ...utils import NumberList, TensorList
from ..functional import sqrt_centered_ema_sq_, sqrt_ema_sq_


def rmsprop_(tensors_: TensorList, exp_avg_sq_: TensorList, smoothing: float | NumberList,
             eps: float | NumberList, debiased: bool, step: int, exp_avg_: TensorList | None = None,
             max_exp_avg_sq_: TensorList | None = None, pow:float=2):
    """returns `tensors_`"""
    if exp_avg_ is not None:
        sqrt_exp_avg_sq = sqrt_centered_ema_sq_(tensors=tensors_, exp_avg_=exp_avg_,
                                                exp_avg_sq_=exp_avg_sq_,max_exp_avg_sq_=max_exp_avg_sq_,
                                                beta=smoothing,debiased=debiased,step=step,pow=pow)
    else:
        sqrt_exp_avg_sq = sqrt_ema_sq_(tensors=tensors_,exp_avg_sq_=exp_avg_sq_,max_exp_avg_sq_=max_exp_avg_sq_,
                                       beta=smoothing,debiased=debiased,step=step,pow=pow)

    return tensors_.div_(sqrt_exp_avg_sq.add_(eps))

class RMSprop(Transform):
    def __init__(self, smoothing: float=0.99, eps:float=1e-8, centered:bool=False, debiased:bool=False,
                 amsgrad:bool=False, pow:float=2,target:Target='update'):
        defaults = dict(smoothing=smoothing,eps=eps,centered=centered,debiased=debiased,amsgrad=amsgrad,pow=pow)
        super().__init__(defaults=defaults, target=target)
        self.current_step = 0

    def transform(self, target, vars):
        self.current_step += 1

        smoothing,eps = self.get_settings('smoothing', 'eps', params=vars, cls=NumberList)
        centered,debiased,amsgrad,pow = itemgetter('centered','debiased','amsgrad','pow')(self.defaults)

        exp_avg_sq = self.get_state('exp_avg_sq', params=vars, cls=TensorList)
        exp_avg = self.get_state('exp_avg', params=vars, cls=TensorList) if centered else None
        max_exp_avg_sq = self.get_state('max_exp_avg_sq', params=vars, cls=TensorList) if amsgrad else None

        return rmsprop_(TensorList(target), exp_avg_sq_=exp_avg_sq, smoothing=smoothing, eps=eps, debiased=debiased,
                        step=self.current_step,exp_avg_=exp_avg, max_exp_avg_sq_=max_exp_avg_sq, pow=pow)