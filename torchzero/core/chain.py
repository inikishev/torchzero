from collections.abc import Iterable

from ..utils.python_tools import flatten
from .module import Module, Chainable


class Chain(Module):
    """Chain of modules, mostly used internally"""
    def __init__(self, *modules: Module | Iterable[Module]):
        super().__init__()
        flat_modules: list[Module] = flatten(modules)
        for i, module in enumerate(flat_modules):
            self.set_child(f'module_{i}', module)

    def update(self, var):
        # note here that `update` and `apply` shouldn't be used directly
        # as it will update all modules, and then apply all modules
        # it is used in specific cases like Chain as trust region hessian module
        for i in range(len(self.children)):
            self.children[f'module_{i}'].update(var)
            if var.stop: break
        return var

    def apply(self, var):
        for i in range(len(self.children)):
            var = self.children[f'module_{i}'].apply(var)
            if var.stop: break
        return var

    def step(self, var):
        for i in range(len(self.children)):
            var = self.children[f'module_{i}'].step(var)
            if var.stop: break
        return var

    def __repr__(self):
        s = self.__class__.__name__
        if self.children:
            if s == 'Chain': s = 'C' # to shorten it
            s = f'{s}({", ".join(str(m) for m in self.children.values())})'
        return s

def maybe_chain(*modules: Chainable) -> Module:
    """Returns a single module directly if only one is provided, otherwise wraps them in a :code:`Chain`."""
    flat_modules: list[Module] = flatten(modules)
    if len(flat_modules) == 1:
        return flat_modules[0]
    return Chain(*flat_modules)


