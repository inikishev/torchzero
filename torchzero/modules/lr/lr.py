import torch

from ...core import Transform
from ...utils import NumberList, TensorList, generic_eq


def lazy_lr(tensors: TensorList, lr: float | list, inplace:bool):
    """multiplies by lr if lr is not 1"""
    if generic_eq(lr, 1): return tensors
    if inplace: return tensors.mul_(lr)
    return tensors * lr

class LR(Transform):
    def __init__(self, lr: float):
        defaults=dict(lr=lr)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        return lazy_lr(TensorList(target), lr=self.get_settings('lr', params=params), inplace=True)

class StepSize(Transform):
    """this is exactly the same as LR, except the `lr` parameter can be renamed to any other name"""
    def __init__(self, step_size: float, key = 'step_size'):
        defaults={"key": key, key: step_size}
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        lrs = []
        for p in params:
            settings = self.settings[p]
            lrs.append(settings[settings['key']])
        return lazy_lr(TensorList(target), lr=lrs, inplace=True)


def warmup(step: int, start_lr: float | NumberList, end_lr: float | NumberList, steps: float):
    """returns warm up lr scalar"""
    if step > steps: return end_lr
    return start_lr + (end_lr - start_lr) * (step / steps)

class Warmup(Transform):
    def __init__(self, start_lr = 1e-5, end_lr = 1, steps = 100):
        defaults = dict(start_lr=start_lr,end_lr=end_lr, steps=steps)
        super().__init__(defaults, uses_grad=False)
        self.global_state['current_step'] = 0

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        start_lr, end_lr = self.get_settings('start_lr', 'end_lr', params=params, cls = NumberList)
        steps = self.settings[params[0]]['steps']
        target = lazy_lr(
            TensorList(target),
            lr=warmup(step=self.global_state['current_step'], start_lr=start_lr, end_lr=end_lr, steps=steps),
            inplace=True
        )
        self.global_state['current_step'] += 1
        return target
