from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

def _adagrad_step_(ascent: TensorList, grad_sum: TensorList, alpha: TensorList, lr_decay: TensorList, eps: TensorList, step: int):
    clr = alpha / (1 + step * lr_decay)
    grad_sum.addcmul_(ascent, ascent)
    return ascent.div_(grad_sum.sqrt().add_(eps)).mul_(clr)

class Adagrad(OptimizerModule):
    """
    Divides ascent direction by mean square root of the sum of all past ascent directions.

    Exactly matches `torch.optim.Adagrad`.

    Args:
        lr_decay (float, optional): learning rate decay. Defaults to 0.
        initial_accumulator_value (float, optional): initial value of the sum of squares of gradients. Defaults to 0.
        eps (float, optional): term added to the denominator to improve numerical stability. Defaults to 1e-10.
        alpha (float, optional): learning rate. Defaults to 1.

    reference
        https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    def __init__(self, lr_decay: float = 0, initial_accumulator_value: float = 0, eps: float = 1e-10, alpha: float = 1):
        defaults = dict(alpha = alpha, lr_decay = lr_decay, initial_accumulator_value=initial_accumulator_value, eps = eps)
        super().__init__(defaults)
        self.cur_step = 0

    @torch.no_grad
    def _update(self, vars, ascent):
        settings = self.get_all_group_keys()
        if self.cur_step == 0: init = ascent.full_like(settings['initial_accumulator_value'])
        else: init = None
        grad_sum = self.get_state_key('grad_sum', init = init) # type:ignore

        updated_direction = _adagrad_step_(
            ascent=ascent,
            grad_sum=grad_sum,
            alpha=settings["alpha"],
            eps=settings["eps"],
            lr_decay=settings["lr_decay"],
            step=self.cur_step,
        )
        self.cur_step += 1
        return updated_direction
