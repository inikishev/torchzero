from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

def _adam_step(ascent: TensorList, exp_avg: TensorList, exp_avg_sq: TensorList, alpha, beta1, beta2, eps, step:int, max_exp_avg_sqs: TensorList | None, params: TensorList | None = None):
    # Decay the first and second moment running average coefficient
    exp_avg.lerp_compat_(ascent, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(ascent, ascent.conj(), value=1 - beta2)

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    if max_exp_avg_sqs is not None:
        max_exp_avg_sqs.maximum_(exp_avg_sq)
        denom = max_exp_avg_sqs.sqrt().div_(bias_correction2**0.5).add_(eps)
    else:
        denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)

    if params is None:
        return (exp_avg / denom).mul_(alpha / bias_correction1)

    # else directly apply the update to params
    params.addcdiv_(exp_avg, denom, value = -(alpha / bias_correction1))
    return None



class Adam(OptimizerModule):
    """Adam. Combines momentum and RMSProp. Exactly matches `torch.optim.Adam`.

    Args:
        beta1 (float, optional): exponential decay rate of gradient moving average. Defaults to 0.9.
        beta2 (float, optional): exponential decay rate of squared gradient moving average. Defaults to 0.999.
        eps (float, optional): epsilon for numerical stability. Defaults to 1e-8.
        amsgrad (bool, optional):
            whether to use the AMSGrad variant of this algorithm from
            On the Convergence of Adam and Beyond (default: False).
        alpha (float, optional): learning rate. Defaults to 1.
    """
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, alpha: float = 1, amsgrad=False):
        defaults = dict(alpha = alpha, beta1=beta1, beta2=beta2, eps=eps)
        super().__init__(defaults)

        self.cur_step = 1
        self.amsgrad = amsgrad

    @torch.no_grad
    def step(self, vars):
        # Adam step is a bit differet from other optimizer steps
        # due to how common it is, I implemented two additional optimizations,

        # 1st - if next module is None or if next module is LR and module after is None
        # this will directly update parameters using `addcdiv_`

        # 2nd - if next module is LR`, adam will "fuse" with it to avoid an additional add operation.

        # the optimizations are quite verbose and seem to barely have any effect, so I probably won't implement
        # this for other modules

        settings = self.get_all_group_keys()

        if self.amsgrad:
            exp_avg, exp_avg_sq, max_exp_avg_sqs = self.get_state_keys('exp_avg', 'exp_avg_sq', 'max_exp_avg_sqs')
        else:
            exp_avg, exp_avg_sq = self.get_state_keys('exp_avg', 'exp_avg_sq')
            max_exp_avg_sqs = None

        params = None

        # apply addcdiv if next module is None
        if self.next_module is None: params = self.get_params()

        # fuse with LR module if it is next
        if self.next_module is not None and self.next_module.IS_LR_MODULE:
            alpha = self.next_module.get_group_key('lr') * settings['alpha']
            self.next_module._skip = True # type:ignore

            # apply addcdiv if module after LR is None.
            if self.next_module.next_module is None: params = self.get_params()

        else:
            alpha = settings['alpha']

        # get params if ascent is None so we need params to access their gradient as initial ascent
        if vars.ascent is None:
            if params is None: pg = self.get_params()
            else: pg = params
        else:
            pg = None

        ret = _adam_step(
            ascent=vars.maybe_use_grad_(pg),
            exp_avg = exp_avg,
            exp_avg_sq = exp_avg_sq,
            alpha = alpha,
            beta1 = settings['beta1'],
            beta2 = settings['beta2'],
            eps = settings['eps'],
            step = self.cur_step,
            max_exp_avg_sqs = max_exp_avg_sqs,
            params = params
        )

        self.cur_step += 1
        if params is None:
            assert ret is not None
            vars.ascent = ret
            return self._update_params_or_step_with_next(vars)

        # next module is either None or LR
        if self.next_module is None: return vars.get_loss()

        # step with LR, which has _skip = True so it won't apply lr, but may step with the scheduler
        self.next_module._update(vars, None) # type:ignore
        return vars.get_loss()