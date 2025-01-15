from collections import abc
import warnings
import torch

from ..core import OptimizerModule, TensorListOptimizer, OptimizationState, _Chain, _Chainable
from ..utils.python_tools import flatten

def _unroll_modules(flat_modules: list[OptimizerModule]) -> list[OptimizerModule]:
    unrolled = []
    for m in flat_modules:
        unrolled.append(m)
        if len(m.children) > 0:
            unrolled.extend(_unroll_modules(list(m.children.values())))
    return unrolled


class Modular(TensorListOptimizer):
    """Creates a modular optimizer by chaining together a sequence of optimizer modules.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        *modules (Iterable[OptimizerModule] | OptimizerModule):
            A sequence of optimizer modules to chain together. This argument will be flattened."""
    def __init__(self, params, *modules: _Chainable):
        flat_modules = flatten(modules)
        self.modules: list[OptimizerModule] = flat_modules
        self.chain = _Chain(flat_modules)

        self.unrolled_modules = _unroll_modules(flat_modules)
        if len([m for m in self.unrolled_modules if m.IS_LR_MODULE]) > 1:
            warnings.warn(
                f'More then 1 lr modules have been added.\
                This may lead to incorrect behaviour with learning rate scheduling and per-parameter learning rates.\
                Make sure there is a single `LR` module, use `Mul` module instead of it where needed.\
                \nList of modules: {self.unrolled_modules}; \nlist of lr modules: {[m for m in self.unrolled_modules if m.IS_LR_MODULE]}'
            )

        if isinstance(params, torch.nn.Module):
            self.model = params
            params = list(params.parameters())
        else:
            self.model = None
            params = list(params)

        super().__init__(params, {})
        self.chain._initialize_(params, set_passed_params=True)

    def get_lr_module(self, last=True) -> OptimizerModule:
        """
        Retrieves the module in the chain that controls the learning rate.

        This method is useful for setting up a learning rate scheduler. By default, it retrieves the last module in the chain
        that has an `lr` group parameter.

        Args:
            last (bool, optional):
                If multiple modules have an `lr` parameter, this argument controls which one is returned.
                - If `True` (default), the last module is returned.
                - If `False`, the first module is returned.

        Returns:
            OptimizerModule: The module that controls the learning rate.

        Raises:
            ValueError: If no modules in the chain have an `lr` parameter. To fix this, add an `LR` module.

        Example:

        .. code:: py
            from torch.optim.lr_scheduler import OneCycleLR
            import torchzero as tz

            opt = tz.Modular(model.parameters(), [tz.m.RMSProp(), tz.m.LR(1e-2), tz.m.DirectionalNewton()])
            lr_scheduler = OneCycleLR(opt.get_lr_module(), max_lr = 1e-1, total_steps = 1000, cycle_momentum=False)

        """
        modules = list(reversed(self.unrolled_modules)) if last else self.unrolled_modules
        for m in modules:
            if 'lr' in m.param_groups[0]: return m

        raise ValueError(f'No modules out of {", ".join(m.__class__.__name__ for m in modules)} support and `lr` parameter. The easiest way to fix is is to add an `LR(1)` module at the end.')

    def get_module_by_name(self, name: str | type, last=True) -> OptimizerModule:
        """Returns the first or last module in the chain that matches the provided name or type.

        Args:
            name (str | type): the name (as a string) or the type of the module to search for.
            last (bool, optional):
                If multiple modules match, this argument controls which one is returned.
                - If `True` (default), the last matching module is returned.
                - If `False`, the first matching module is returned.

        Returns:
            OptimizerModule: The matching optimizer module.

        Raises:
            ValueError: If no modules in the chain match the provided name or type.
        """
        modules = list(reversed(self.unrolled_modules)) if last else self.unrolled_modules
        for m in modules:
            if isinstance(name, str) and m.__class__.__name__ == name: return m
            if isinstance(name, type) and isinstance(m, name): return m

        raise ValueError(f'No modules out of {", ".join(m.__class__.__name__ for m in modules)} match "{name}".')

    def step(self, closure=None): # type:ignore
        state = OptimizationState(closure, self.model)
        res = self.chain.step(state)
        for hook in state.post_step_hooks: hook(self, state)
        return res
