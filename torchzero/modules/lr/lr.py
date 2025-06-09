"""Learning rate"""
import torch

from ...core import Transform
from ...utils import NumberList, TensorList, generic_eq, unpack_dicts

def lazy_lr(tensors: TensorList, lr: float | list, inplace:bool):
    """multiplies by lr if lr is not 1"""
    if generic_eq(lr, 1): return tensors
    if inplace: return tensors.mul_(lr)
    return tensors * lr

class LR(Transform):
    """Learning rate. Adding this module also adds support for LR schedulers."""
    def __init__(self, lr: float):
        defaults=dict(lr=lr)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        return lazy_lr(TensorList(tensors), lr=[s['lr'] for s in settings], inplace=True)

class StepSize(Transform):
    """this is exactly the same as LR, except the `lr` parameter can be renamed to any other name to avoid clashes"""
    def __init__(self, step_size: float, key = 'step_size'):
        defaults={"key": key, key: step_size}
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        return lazy_lr(TensorList(tensors), lr=[s[s['key']] for s in settings], inplace=True)


def _warmup_lr(step: int, start_lr: float | NumberList, end_lr: float | NumberList, steps: float):
    """returns warm up lr scalar"""
    if step > steps: return end_lr
    return start_lr + (end_lr - start_lr) * (step / steps)

class Warmup(Transform):
    """Learning rate warmup, linearly increases learning rate multiplier from :code:`start_lr` to :code:`end_lr` over :code:`steps` steps.

    Args:
        start_lr (_type_, optional): initial learning rate multiplier on first step. Defaults to 1e-5.
        end_lr (float, optional): learning rate multiplier at the end and after warmup. Defaults to 1.
        steps (int, optional): number of steps to perform warmup for. Defaults to 100.
    """
    def __init__(self, start_lr = 1e-5, end_lr:float = 1, steps = 100):
        defaults = dict(start_lr=start_lr,end_lr=end_lr, steps=steps)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        start_lr, end_lr = unpack_dicts(settings, 'start_lr', 'end_lr', cls = NumberList)
        num_steps = settings[0]['steps']
        step = self.global_state.get('step', 0)

        target = lazy_lr(
            TensorList(tensors),
            lr=_warmup_lr(step=step, start_lr=start_lr, end_lr=end_lr, steps=num_steps),
            inplace=True
        )
        self.global_state['step'] = step + 1
        return target
