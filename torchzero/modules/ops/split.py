from collections.abc import Callable
from typing import cast

import torch

from ...core import Chainable, Module, Vars


def _split(
    module: Module,
    idxs,
    params,
    vars: Vars,
):
    split_params = [p for i,p in enumerate(params) if i in idxs]

    split_grad = None
    if vars.grad is not None:
        split_grad = [g for i,g in enumerate(vars.grad) if i in idxs]

    split_update = None
    if vars.update is not None:
        split_update = [u for i,u in enumerate(vars.update) if i in idxs]

    split_vars = vars.clone(clone_update=False)
    split_vars.params = split_params
    split_vars.grad = split_grad
    split_vars.update = split_update

    split_vars = module.step(split_vars)

    if (vars.grad is None) and (split_vars.grad is not None):
        vars.grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    if split_vars.update is not None:

        if vars.update is None:
            if vars.grad is None: vars.update = [cast(torch.Tensor, None) for _ in vars.params]
            else: vars.update = [g.clone() for g in vars.grad]

        for idx, u in zip(idxs, split_vars.update):
            vars.update[idx] = u

    vars.update_attrs_from_clone_(split_vars)
    return vars

class Split(Module):
    """Apply `true` modules to all parameters filtered by `filter`, apply `false` modules to all other parameters."""
    def __init__(self, filter: Callable[[torch.Tensor], bool], true: Chainable | None, false: Chainable | None):
        defaults = dict(filter=filter)
        super().__init__(defaults)

        if true is not None: self.set_child('true', true)
        if false is not None: self.set_child('false', false)

    def step(self, vars):

        params = vars.params
        filter = self.settings[params[0]]['filter']

        true_idxs = []
        false_idxs = []
        for i,p in enumerate(params):
            if filter(p): true_idxs.append(i)
            else: false_idxs.append(i)

        if 'true' in self.children:
            true = self.children['true']
            vars = _split(true, idxs=true_idxs, params=params, vars=vars)

        if 'false' in self.children:
            false = self.children['false']
            vars = _split(false, idxs=false_idxs, params=params, vars=vars)

        return vars