"""Learning rate"""
import torch
import random

from ...core import TensorTransform
from ...utils import NumberList, TensorList, generic_ne, unpack_dicts

def lazy_lr(tensors: TensorList, lr: float | list, inplace:bool):
    """multiplies by lr if lr is not 1"""
    if generic_ne(lr, 1):
        if inplace: return tensors.mul_(lr)
        return tensors * lr
    return tensors

class LR(TensorTransform):
    """Learning rate. Adding this module also adds support for LR schedulers."""
    def __init__(self, lr: float):
        defaults=dict(lr=lr)
        super().__init__(defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        return lazy_lr(TensorList(tensors), lr=[s['lr'] for s in settings], inplace=True)

class StepSize(TensorTransform):
    """this is exactly the same as LR, except the `lr` parameter can be renamed to any other name to avoid clashes"""
    def __init__(self, step_size: float, key = 'step_size'):
        defaults={"key": key, key: step_size}
        super().__init__(defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        return lazy_lr(TensorList(tensors), lr=[s[s['key']] for s in settings], inplace=True)


def _warmup_lr(step: int, start_lr: float | NumberList, end_lr: float | NumberList, steps: float):
    """returns warm up lr scalar"""
    if step > steps: return end_lr
    return start_lr + (end_lr - start_lr) * (step / steps)

class Warmup(TensorTransform):
    """Learning rate warmup, linearly increases learning rate multiplier from ``start_lr`` to ``end_lr`` over ``steps`` steps.

    Args:
        steps (int, optional): number of steps to perform warmup for. Defaults to 100.
        start_lr (_type_, optional): initial learning rate multiplier on first step. Defaults to 1e-5.
        end_lr (float, optional): learning rate multiplier at the end and after warmup. Defaults to 1.

    Example:
        Adam with 1000 steps warmup

        .. code-block:: python

            opt = tz.Optimizer(
                model.parameters(),
                tz.m.Adam(),
                tz.m.LR(1e-2),
                tz.m.Warmup(steps=1000)
            )

    """
    def __init__(self, steps = 100, start_lr = 1e-5, end_lr:float = 1):
        defaults = dict(start_lr=start_lr,end_lr=end_lr, steps=steps)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        start_lr, end_lr = unpack_dicts(settings, 'start_lr', 'end_lr', cls = NumberList)
        num_steps = settings[0]['steps']
        step = self.global_state.get('step', 0)

        tensors = lazy_lr(
            TensorList(tensors),
            lr=_warmup_lr(step=step, start_lr=start_lr, end_lr=end_lr, steps=num_steps),
            inplace=True
        )
        self.global_state['step'] = step + 1
        return tensors

class WarmupNormClip(TensorTransform):
    """Warmup via clipping of the update norm.

    Args:
        start_norm (_type_, optional): maximal norm on the first step. Defaults to 1e-5.
        end_norm (float, optional): maximal norm on the last step. After that, norm clipping is disabled. Defaults to 1.
        steps (int, optional): number of steps to perform warmup for. Defaults to 100.

    Example:
        Adam with 1000 steps norm clip warmup

        .. code-block:: python

            opt = tz.Optimizer(
                model.parameters(),
                tz.m.Adam(),
                tz.m.WarmupNormClip(steps=1000)
                tz.m.LR(1e-2),
            )
    """
    def __init__(self, steps = 100, start_norm = 1e-5, end_norm:float = 1):
        defaults = dict(start_norm=start_norm,end_norm=end_norm, steps=steps)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        start_norm, end_norm = unpack_dicts(settings, 'start_norm', 'end_norm', cls = NumberList)
        num_steps = settings[0]['steps']
        step = self.global_state.get('step', 0)
        if step > num_steps: return tensors

        tensors = TensorList(tensors)
        norm = tensors.global_vector_norm()
        current_max_norm = _warmup_lr(step, start_norm[0], end_norm[0], num_steps)
        if norm > current_max_norm:
            tensors.mul_(current_max_norm / norm)

        self.global_state['step'] = step + 1
        return tensors


class RandomStepSize(TensorTransform):
    """Uses random global or layer-wise step size from ``low`` to ``high``.

    Args:
        low (float, optional): minimum learning rate. Defaults to 0.
        high (float, optional): maximum learning rate. Defaults to 1.
        parameterwise (bool, optional):
            if True, generate random step size for each parameter separately,
            if False generate one global random step size. Defaults to False.
    """
    def __init__(self, low: float = 0, high: float = 1, parameterwise=False, seed:int|None=None):
        defaults = dict(low=low, high=high, parameterwise=parameterwise,seed=seed)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        s = settings[0]
        parameterwise = s['parameterwise']

        seed = s['seed']
        if 'generator' not in self.global_state:
            self.global_state['generator'] = random.Random(seed)
        generator: random.Random = self.global_state['generator']

        if parameterwise:
            low, high = unpack_dicts(settings, 'low', 'high')
            lr = [generator.uniform(l, h) for l, h in zip(low, high)]
        else:
            low = s['low']
            high = s['high']
            lr = generator.uniform(low, high)

        torch._foreach_mul_(tensors, lr)
        return tensors
