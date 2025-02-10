from collections.abc import Callable, Iterable
import numpy as np
import torch

from ...core import OptimizerModule

_Value = int | float | OptimizerModule | Iterable[OptimizerModule]


class Sum(OptimizerModule):
    """calculates sum of multiple updates.

    Args:
        *modules:
            either OptimizerModules or iterables of OptimizerModules to chain. Scalars are also allowed."""
    def __init__(
        self,
        *modules: _Value,
    ):
        super().__init__({})

        scalars = [i for i in modules if isinstance(i, (int,float))]
        self.scalar = sum(scalars) if len(scalars) > 0 else None

        for i,module in enumerate(i for i in modules if not isinstance(i, (int, float))):
            self._set_child_(i, module)

    @torch.no_grad
    def step(self, vars):
        if len(self.children) == 1:
            vars.ascent = self.children[0].return_ascent(vars)
            if self.scalar is not None: vars.ascent += self.scalar
            return self._update_params_or_step_with_next(vars)

        sum = None
        for i, c in sorted(self.children.items(), key=lambda x: x[0]):
            if i == len(self.children) - 1: cur_state = vars
            else: cur_state = vars.copy(clone_ascent = True)

            if sum is None: sum = c.return_ascent(cur_state)
            else: sum += c.return_ascent(cur_state)

            if i != len(self.children) - 1: vars.update_attrs_(cur_state)

        assert sum is not None
        if self.scalar is not None: sum += self.scalar
        vars.ascent = sum
        return self._update_params_or_step_with_next(vars)

class Mean(OptimizerModule):
    """calculates mean of multiple updates.

    Args:
        *modules:
            either OptimizerModules or iterables of OptimizerModules to chain. Scalars are also allowed."""

    def __init__(
        self,
        *modules: _Value,
    ):
        super().__init__({})

        scalars = [i for i in modules if isinstance(i, (int,float))]
        self.scalar = sum(scalars) if len(scalars) > 0 else None

        self.n_values = len(modules)

        for i,module in enumerate(i for i in modules if not isinstance(i, (int, float))):
            self._set_child_(i, module)

    @torch.no_grad
    def step(self, vars):
        if len(self.children) == 1:
            vars.ascent = self.children[0].return_ascent(vars)
            if self.scalar is not None: vars.ascent += self.scalar
            if self.n_values > 1: vars.ascent /= self.n_values
            return self._update_params_or_step_with_next(vars)

        sum = None
        for i, c in sorted(self.children.items(), key=lambda x: x[0]):
            if i == len(self.children) - 1: cur_state = vars
            else: cur_state = vars.copy(clone_ascent = True)

            if sum is None: sum = c.return_ascent(cur_state)
            else: sum += c.return_ascent(cur_state)

            if i != len(self.children) - 1: vars.update_attrs_(cur_state)

        assert sum is not None
        if self.scalar is not None: sum += self.scalar
        if self.n_values > 1: sum /= self.n_values
        vars.ascent = sum
        return self._update_params_or_step_with_next(vars)

class Product(OptimizerModule):
    """calculates product of multiple updates.

    Args:
        *modules:
            either OptimizerModules or iterables of OptimizerModules to chain. Scalars are also allowed."""

    def __init__(
        self,
        *modules: _Value,
    ):
        super().__init__({})

        scalars = [i for i in modules if isinstance(i, (int,float))]
        self.scalar = np.prod(scalars).item() if len(scalars) > 0 else None

        for i,module in enumerate(i for i in modules if not isinstance(i, (int, float))):
            self._set_child_(i, module)

    @torch.no_grad
    def step(self, vars):
        if len(self.children) == 1:
            vars.ascent = self.children[0].return_ascent(vars)
            if self.scalar is not None: vars.ascent *= self.scalar
            return self._update_params_or_step_with_next(vars)

        prod = None
        for i, c in sorted(self.children.items(), key=lambda x: x[0]):
            if i == len(self.children) - 1: cur_state = vars
            else: cur_state = vars.copy(clone_ascent = True)

            if prod is None: prod = c.return_ascent(cur_state)
            else: prod *= c.return_ascent(cur_state)

            if i != len(self.children) - 1: vars.update_attrs_(cur_state)

        assert prod is not None
        if self.scalar is not None: prod *= self.scalar
        vars.ascent = prod
        return self._update_params_or_step_with_next(vars)
