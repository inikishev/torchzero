from collections import abc
import torch

from ..core import OptimizerModule, TensorListOptimizer, OptimizationState, _Chain, _Chainable
from ..utils.python_tools import flatten

class Modular(TensorListOptimizer):
    """Make a modular optimizer from a sequence of modules.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        modules (Iterable[OptimizerModule] | OptimizerModule):
            sequence of modules to chain together. Any passed sequence will be flattened.
    """
    def __init__(self, params, *modules: _Chainable):
        flat_modules = flatten(modules)
        self.modules: list[OptimizerModule] = flat_modules
        self.chain = _Chain(flat_modules)

        if isinstance(params, torch.nn.Module):
            self.model = params
            params = list(params.parameters())
        else:
            self.model = None
            params = list(params)

        super().__init__(params, {})
        self.chain._initialize_(params)

    def get_lr_module(self, last=True) -> OptimizerModule:
        """returns the last module with `lr` parameter in it, which can be passed to a pytorch learning rate scheduler.
        If no module has `lr` setting, this will raise ValueError.

        Notes:
            - If learning rate is specified in a module in the middle, for example `[Normalize(), LR(1e-2), Adam()]`,\
            here this will pick `Adam` if last=True, or `Normalize` if last=False. Both have default lr of s1, so if\
            scheduled, the scheduler lr will be multiplied by 1e-2. All optimizers in `torchzero.optim` are guaranteed to\
            have the lr module last.

            - This will not find nested modules like `[Graft(Adam(), SGD())]`. Either add `LR` after `Graft`, or create `Adam`
            separately beforehand and pass it to the scheduler after creating the modular optimizer.

        Args:
            last (bool, optional):
                if multiple modules support `lr`, if True return last one, otherwise returns first.
                Usually you would want the last one as first ones might get overriden by things like Normalize(). Defaults to True.

        example:

        .. code:: py
            from torch.optim.lr_scheduler import OneCycleLR
            import torchzero as tz

            opt = tz.Modular(model.parameters(), [tz.m.RMSProp(), tz.m.LR(1e-2)])
            lr_scheduler = OneCycleLR(opt.get_lr_module(), max_lr = 1e-1, total_steps = 1000, cycle_momentum=False)

        """
        modules = list(reversed(self.modules)) if last else self.modules
        for m in modules:
            if 'lr' in m.param_groups[0]: return m

        raise ValueError(f'No modules out of {", ".join(m.__class__.__name__ for m in modules)} support and `lr` parameter. The easiest way to fix is is to add an `LR(1)` module at the end.')

    def get_module_by_name(self, name: str | type, last=True) -> OptimizerModule:
        """Returns first or last module which class name matches `name`, or whose class is `name` if it is a type.
        If no module found, raises ValueError

        Args:
            name (str | type): name to match, or class.
            last (bool, optional): if multiple modules match, if True return last one, otherwise returns first. Defaults to True.
        """
        modules = list(reversed(self.modules)) if last else self.modules
        for m in modules:
            if isinstance(name, str) and m.__class__.__name__ == name: return m
            if isinstance(name, type) and isinstance(m, name): return m

        raise ValueError(f'No modules out of {", ".join(m.__class__.__name__ for m in modules)} match "{name}".')

    def step(self, closure=None): # type:ignore
        state = OptimizationState(closure, self.model)
        res = self.chain.step(state)
        for hook in state.post_step_hooks: hook(self, state)
        return res
