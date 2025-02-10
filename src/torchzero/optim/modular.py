from collections import abc
import warnings
from inspect import cleandoc
import torch

from ..core import OptimizerModule, TensorListOptimizer, OptimizationVars, _Chain, _Chainable
from ..utils.python_tools import flatten

def _unroll_modules(flat_modules: list[OptimizerModule], nested) -> list[OptimizerModule]:
    """returns a list of all modules, including all nested ones"""
    unrolled = []
    for m in flat_modules:
        unrolled.append(m)
        if len(m.children) > 0:
            unrolled.extend(_unroll_modules(list(m.children.values()), nested=True))
        if nested:
            if m.next_module is not None:
                unrolled.extend(_unroll_modules([m.next_module], nested=True))
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

        # save unrolled modules and make sure there is only 1 LR module.
        self.unrolled_modules = _unroll_modules(flat_modules, nested=False)
        num_lr_modules = len([m for m in self.unrolled_modules if m.IS_LR_MODULE])
        if num_lr_modules > 1:
            warnings.warn(cleandoc(
                f"""More then 1 lr modules have been added.
                This may lead to incorrect behaviour with learning rate scheduling and per-parameter learning rates.
                Make sure there is a single `LR` module, use `Alpha` module instead of it where needed.
                \nList of modules: {self.unrolled_modules}; \nlist of lr modules: {[m for m in self.unrolled_modules if m.IS_LR_MODULE]}"""
            ))

        if isinstance(params, torch.nn.Module):
            self.model = params
            params = list(params.parameters())
        else:
            self.model = None
            params = list(params)

        # if there is an `lr` setting, make sure there is an LR module that can use it
        for p in params:
            if isinstance(p, dict):
                if 'lr' in p:
                    if num_lr_modules == 0:
                        warnings.warn(cleandoc(
                            """Passed "lr" setting in a parameter group, but there is no LR module that can use that setting.
                            Add an `LR` module to make per-layer "lr" setting work."""
                        ))

        super().__init__(params, {})
        self.chain._initialize_(params, set_passed_params=True)

        # run post-init hooks
        for module in self.unrolled_modules:
            for hook in module.post_init_hooks:
                hook(self, module)

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
        vars = OptimizationVars(closure, self.model)
        res = self.chain.step(vars)
        for hook in vars.post_step_hooks: hook(self, vars)
        return res
