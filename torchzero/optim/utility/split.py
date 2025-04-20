import warnings
from collections.abc import Callable, Iterable

import torch

from ...utils import flatten, get_params

class Split(torch.optim.Optimizer):
    """Steps will all `optimizers`, also has a check that they have no duplicate parameters.
    Doesn't support closure based optimizers.

    Example:

    .. code:: py

        opt = Split(
            torch.optim.Adam(model.encoder.parameters(), lr=0.001),
            torch.optim.SGD(model.decoder.parameters(), lr=0.1)
        )
    """
    def __init__(self, *optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer]):
        all_params = []
        self.optimizers: list[torch.optim.Optimizer] = flatten(optimizers)

        # gather all params in case user tries to access them from this object
        for i,opt in enumerate(self.optimizers):
            for p in get_params(opt.param_groups, 'all', list):
                if p not in all_params: all_params.append(p)
                else: warnings.warn(
                    f'optimizers[{i}] {opt.__class__.__name__} has some duplicate parameters '
                    'that are also in previous optimizers. They will be updated multiple times.')

        super().__init__(all_params, {})

    def step(self, closure: Callable | None = None):
        loss = None

        # if closure provided, populate grad, otherwise each optimizer will call closure separately
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for opt in self.optimizers:
            opt.step() # closure not passed as grad is already evaluated

        return loss