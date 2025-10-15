from collections.abc import Mapping, Sequence, Iterable, Callable
from typing import TYPE_CHECKING, Any

import torch

from .objective import Objective

if TYPE_CHECKING:
    from .module import Module
    from .transform import Transform



def update(
    objective: "Objective",
    module: "Transform",
    states: list[dict[str, Any]] | None = None,
    settings: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    if states is None:
        assert settings is None
        module.update(objective)

    else:
        assert settings is not None
        module.update_states(objective, states, settings)

def apply(
    objective: "Objective",
    module: "Transform",
    states: list[dict[str, Any]] | None = None,
    settings: Sequence[Mapping[str, Any]] | None = None,
) -> "Objective":
    if states is None:
        assert settings is None
        return module.apply(objective)

    else:
        assert settings is not None
        return module.apply_states(objective, states, settings)

def _chain_step(objective: "Objective", modules: "Sequence[Module]"):
    """steps with ``modules`` and returns updated objective, this is used within ``step`` and within ``Chain.step``"""
    # step
    for i, module in enumerate(modules):
        if i!=0: objective = objective.clone(clone_updates=False)

        objective = module.step(objective)
        if objective.stop: break

    return objective

def step(objective: "Objective", modules: "Module | Sequence[Module]"):
    """doesn't apply hooks!"""
    if not isinstance(modules, Sequence):
        modules = (modules, )

    if len(modules) == 0:
        raise RuntimeError("`modules` is an empty sequence")

    # if closure is None, assume backward has been called and gather grads
    if objective.closure is None:
        objective.grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in objective.params]

    # step and return
    return _chain_step(objective, modules)


def step_tensors(
    modules: "Module | Sequence[Module]",
    tensors: Sequence[torch.Tensor],
    params: Iterable[torch.Tensor] | None = None,
    grads: Sequence[torch.Tensor] | None = None,
    loss: torch.Tensor | None = None,
    closure: Callable | None = None,
    objective: "Objective | None" = None
) -> list[torch.Tensor]:
    if objective is not None:
        if any(i is not None for i in (params, grads, loss, closure)):
            raise RuntimeError("Specify either `objective` or `(params, grads, loss, closure)`")

    if not isinstance(modules, Sequence):
        modules = (modules, )

    # make fake params if they are only used for shapes
    # note that if modules use states, tensors must always be the same python object
    if params is None:
        params = [t.view_as(t).requires_grad_() for t in tensors]

    # create objective
    if objective is None:
        objective = Objective(params=params, loss=loss, closure=closure)

    if grads is not None:
        objective.grads = list(grads)

    objective.updates = list(tensors)

    # step with modules
    # this won't update parameters in-place (on modules with fused update) because objective.Optimizer is None
    objective = _chain_step(objective, modules)

    # return updates
    return objective.get_updates()