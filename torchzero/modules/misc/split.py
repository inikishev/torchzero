from collections.abc import Callable, Sequence, Iterable
from typing import cast

import torch

from ...core import Chainable, Module, Objective


def _split(
    module: Module,
    idxs,
    params,
    objective: Objective,
):
    split_params = [p for i,p in enumerate(params) if i in idxs]

    split_grad = None
    if objective.grads is not None:
        split_grad = [g for i,g in enumerate(objective.grads) if i in idxs]

    split_update = None
    if objective.updates is not None:
        split_update = [u for i,u in enumerate(objective.updates) if i in idxs]

    split_obj = objective.clone(clone_updates=False, parent=objective)
    split_obj.params = split_params
    split_obj.grads = split_grad
    split_obj.updates = split_update

    split_obj = module.step(split_obj)

    # those should be set due to var being parent
    if split_obj.grads is not None:
        assert objective.grads is not None

    if split_obj.loss is not None:
        assert objective.loss is not None

    if split_obj.updates is not None:

        # make sure update is set, it will be filled with ``true`` and ``false`` tensors
        if objective.updates is None:
            if objective.grads is None: objective.updates = [cast(torch.Tensor, None) for _ in objective.params]
            else: objective.updates = [g.clone() for g in objective.grads]

        # set all tensors from this split
        for idx, u in zip(idxs, split_obj.updates):
            objective.updates[idx] = u

    return objective

_SingleFilter = Callable[[torch.Tensor], bool] | torch.Tensor | Iterable[torch.Tensor] | torch.nn.Module | Iterable[torch.nn.Module]
Filter = _SingleFilter | Iterable[_SingleFilter]

def _make_filter(filter: Filter):
    if callable(filter): return filter
    if isinstance(filter, torch.Tensor):
        return lambda x: x is filter
    if isinstance(filter, torch.nn.Module):
        return _make_filter(filter.parameters())

    # iterable
    filters = [_make_filter(f) for f in filter]
    return lambda x: any(f(x) for f in filters)

class Split(Module):
    """Apply ``true`` modules to all parameters filtered by ``filter``, apply ``false`` modules to all other parameters.

    Args:
        filter (Filter, bool]):
            a filter that selects tensors to be optimized by ``true``.
            - tensor or iterable of tensors (e.g. ``encoder.parameters()``).
            - function that takes in tensor and outputs a bool (e.g. ``lambda x: x.ndim >= 2``).
            - a sequence of above (acts as "or", so returns true if any of them is true).

        true (Chainable | None): modules that are applied to tensors where ``filter`` is ``True``.
        false (Chainable | None): modules that are applied to tensors where ``filter`` is ``False``.

    ### Examples:

    Muon with Adam fallback using same hyperparams as https://github.com/KellerJordan/Muon

    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.NAG(0.95),
        tz.m.Split(
            lambda p: p.ndim >= 2,
            true = tz.m.Orthogonalize(),
            false = [tz.m.Adam(0.9, 0.95), tz.m.Mul(1/66)],
        ),
        tz.m.LR(1e-2),
    )
    ```
    """
    def __init__(self, filter: Filter, true: Chainable | None, false: Chainable | None):
        defaults = dict(filter=filter)
        super().__init__(defaults)

        if true is not None: self.set_child('true', true)
        if false is not None: self.set_child('false', false)

    def update(self, objective): raise RuntimeError
    def apply(self, objective): raise RuntimeError

    def step(self, objective):

        params = objective.params
        filter = _make_filter(self.settings[params[0]]['filter'])

        true_idxs = []
        false_idxs = []
        for i,p in enumerate(params):
            if filter(p): true_idxs.append(i)
            else: false_idxs.append(i)

        if 'true' in self.children and len(true_idxs) > 0:
            true = self.children['true']
            objective = _split(true, idxs=true_idxs, params=params, objective=objective)

        if 'false' in self.children and len(false_idxs) > 0:
            false = self.children['false']
            objective = _split(false, idxs=false_idxs, params=params, objective=objective)

        return objective