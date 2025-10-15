from collections.abc import Iterable

from ..utils.python_tools import flatten
from .functional import _chain_step
from .module import Chainable, Module


class Chain(Module):
    """Chain modules, mostly used internally"""
    def __init__(self, *modules: Module | Iterable[Module]):
        super().__init__()
        flat_modules: list[Module] = flatten(modules)
        for i, module in enumerate(flat_modules):
            self.set_child(f'module_{i}', module)

    def update(self, objective):
        if len(self.children) > 1:
            raise RuntimeError("can't call `update` on Chain with more than one child, as `update` and `apply` have to be called sequentially. Use the `step` method instead of update-apply.")

        if len(self.children) == 0: return
        return self.children['module_0'].update(objective)

    def apply(self, objective):
        if len(self.children) > 1:
            raise RuntimeError("can't call `update` on Chain with more than one child, as `update` and `apply` have to be called sequentially. Use the `step` method instead of update-apply.")

        if len(self.children) == 0: return objective
        return self.children['module_0'].apply(objective)

    def step(self, objective):
        children = [self.children[f'module_{i}'] for i in range(len(self.children))]
        return _chain_step(objective, children)

    def __repr__(self):
        s = self.__class__.__name__
        if self.children:
            if s == 'Chain': s = 'C' # to shorten it
            s = f'{s}({", ".join(str(m) for m in self.children.values())})'
        return s

def maybe_chain(*modules: Chainable) -> Module:
    """Returns a single module directly if only one is provided, otherwise wraps them in a ``Chain``."""
    flat_modules: list[Module] = flatten(modules)
    if len(flat_modules) == 1:
        return flat_modules[0]
    return Chain(*flat_modules)


