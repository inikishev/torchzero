from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .module import Module
    from .objective import Objective



def update(
    objective: "Objective",
    module: "Module",
    states: list[dict[str, Any]] | None = None,
    settings: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    if states is None:
        assert settings is None
        module.update_internal(objective)

    else:
        assert settings is not None
        module.update(objective, states, settings)

def apply(
    objective: "Objective",
    module: "Module",
    states: list[dict[str, Any]] | None = None,
    settings: Sequence[Mapping[str, Any]] | None = None,
) -> "Objective":
    if states is None:
        assert settings is None
        return module.apply_internal(objective)

    assert settings is not None
    return module.apply(objective, states, settings)


def step(objective: "Objective", modules: "Module | Sequence[Module]"):
    if not isinstance(modules, Sequence):
        modules = (modules, )

    if len(modules) == 0:
        raise RuntimeError("`modules` is an empty sequence")

    # if closure is None, assume backward has been called and gather grads
    if objective.closure is None:
        objective.grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in objective.params]

    # step
    for i, module in enumerate(modules):
        if i!=0: objective = objective.clone(clone_update=False)

        module.update_internal(objective)
        objective = module.apply_internal(objective)

        if objective.stop: break

    # apply hooks
    for hook in objective.post_step_hooks:
        hook(objective, tuple(modules))

    return objective
