from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .module import Module
    from .var import Var


def step(var: "Var", modules: "Sequence[Module]",) -> "Var":
    """steps with ``modules`` and returns modified ``var``, doesn't update parameters.

    Args:
        var (Var): Var object.
        modules (Sequence[Module]): sequence of modules to step with.

    Returns:
        Var: modified Var
    """
    # step
    for i, module in enumerate(modules):
        if i!=0: var = var.clone(clone_update=False)

        ret = module.update(var)
        var = module.apply(var, ret)

        if var.stop: break

    return var
