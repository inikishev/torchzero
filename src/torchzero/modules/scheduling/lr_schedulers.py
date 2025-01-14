import warnings, random
from ...core import OptimizerModule

class LRWarmup(OptimizerModule):
    """Linear learning rate warmup.

    Args:
        n_steps (int): number of warmup steps.
        start_lr (float, optional): initial lr. Defaults to 1e-8.
        end_lr (float, optional): final lr. Defaults to 1.
        delay_steps (int, optional): number of `start_lr` steps before starting the warmup. Defaults to 0.
    """
    def __init__(self, n_steps: int, start_lr: float = 1e-8, end_lr: float = 1, delay_steps: int = 0):

        super().__init__({})
        self.n_steps = n_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.delay_steps = delay_steps

        self.cur = 0

    def _update(self, state, ascent):
        if self.cur < self.delay_steps:
            if self.start_lr != 1: ascent *= self.start_lr

        elif self.cur >= self.n_steps + self.delay_steps:
            if self.end_lr != 1: ascent *= self.end_lr

        else:
            remaining = (self.n_steps - (self.cur-self.delay_steps)) / self.n_steps
            lr = (self.start_lr * remaining) + self.end_lr * (1 - remaining)
            ascent *= lr

        self.cur += 1
        return ascent


