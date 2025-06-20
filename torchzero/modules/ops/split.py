from collections.abc import Callable
from typing import cast

import torch

from ...core import Chainable, Module, Var


def _split(
    module: Module,
    idxs,
    params,
    var: Var,
):
    split_params = [p for i,p in enumerate(params) if i in idxs]

    split_grad = None
    if var.grad is not None:
        split_grad = [g for i,g in enumerate(var.grad) if i in idxs]

    split_update = None
    if var.update is not None:
        split_update = [u for i,u in enumerate(var.update) if i in idxs]

    split_var = var.clone(clone_update=False)
    split_var.params = split_params
    split_var.grad = split_grad
    split_var.update = split_update

    split_var = module.step(split_var)

    if (var.grad is None) and (split_var.grad is not None):
        var.grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    if split_var.update is not None:

        if var.update is None:
            if var.grad is None: var.update = [cast(torch.Tensor, None) for _ in var.params]
            else: var.update = [g.clone() for g in var.grad]

        for idx, u in zip(idxs, split_var.update):
            var.update[idx] = u

    var.update_attrs_from_clone_(split_var)
    return var

class Split(Module):
    """Apply `true` modules to all parameters filtered by `filter`, apply `false` modules to all other parameters.

    Args:
        filter (Callable[[torch.Tensor], bool]): a function that takes in a parameter tensor and returns a boolean value.
        true (Chainable | None): modules that are applied to tensors where :code:`filter` returned True.
        false (Chainable | None): modules that are applied to tensors where :code:`filter` returned False.

    Examples:
        standard Muon with Adam fallback

        .. code-block:: python

            opt = tz.Modular(
                model.head.parameters(),
                tz.m.Split(
                    # apply muon only to 2D+ parameters
                    filter = lambda t: t.ndim >= 2,
                    true = [
                        tz.m.HeavyBall(),
                        tz.m.Orthogonalize(),
                        tz.m.LR(1e-2),
                    ],
                    false = tz.m.Adam()
                ),
                tz.m.LR(1e-2)
            )


    """
    def __init__(self, filter: Callable[[torch.Tensor], bool], true: Chainable | None, false: Chainable | None):
        defaults = dict(filter=filter)
        super().__init__(defaults)

        if true is not None: self.set_child('true', true)
        if false is not None: self.set_child('false', false)

    def step(self, var):

        params = var.params
        filter = self.settings[params[0]]['filter']

        true_idxs = []
        false_idxs = []
        for i,p in enumerate(params):
            if filter(p): true_idxs.append(i)
            else: false_idxs.append(i)

        if 'true' in self.children:
            true = self.children['true']
            var = _split(true, idxs=true_idxs, params=params, var=var)

        if 'false' in self.children:
            false = self.children['false']
            var = _split(false, idxs=false_idxs, params=params, var=var)

        return var