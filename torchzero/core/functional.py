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
    # n_modules = len(modules)
    # if n_modules == 0: return var.clone(clone_update=False)
    # last_module = modules[-1]
    # last_lr = last_module.defaults.get('lr', None)

    # step
    for i, module in enumerate(modules):
        if i!=0: var = var.clone(clone_update=False)

        # last module, or next to last module before lr
        # if (i == n_modules - 1) or ((i == n_modules - 2) and (last_lr is not None)):
        #     if len(module.children) != 0 or is_nested: var.nested_is_last = True
        #     else: var.is_last = True
        #     if last_lr is not None: var.last_module_lrs = [last_module.settings[p]['lr'] for p in var.params]

        var = module.step(var)
        if var.stop: break

    return var
