from collections.abc import Callable, Iterable
from operator import methodcaller
from typing import cast, overload, Any
import numpy as np
import torch

from ...core import OptimizerModule
from ...tensorlist import TensorList
from ..meta.chain import Chain

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
            self._set_child_(i, Chain(module))

    @torch.no_grad
    def step(self, state):
        if len(self.children) == 1:
            state.ascent = self.children[0].return_ascent(state)
            if self.scalar is not None: state.ascent += self.scalar
            return self._update_params_or_step_with_next(state)

        sum = None
        for i, c in sorted(self.children.items(), key=lambda x: x[0]):
            if i == len(self.children) - 1: cur_state = state
            else: cur_state = state.copy(clone_ascent = True)

            if sum is None: sum = c.return_ascent(cur_state)
            else: sum += c.return_ascent(cur_state)

            if i != len(self.children) - 1: state.update_attrs_(cur_state)

        assert sum is not None
        if self.scalar is not None: sum += self.scalar
        state.ascent = sum
        return self._update_params_or_step_with_next(state)

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
            self._set_child_(i, Chain(module))

    @torch.no_grad
    def step(self, state):
        if len(self.children) == 1:
            state.ascent = self.children[0].return_ascent(state)
            if self.scalar is not None: state.ascent += self.scalar
            if self.n_values > 1: state.ascent /= self.n_values
            return self._update_params_or_step_with_next(state)

        sum = None
        for i, c in sorted(self.children.items(), key=lambda x: x[0]):
            if i == len(self.children) - 1: cur_state = state
            else: cur_state = state.copy(clone_ascent = True)

            if sum is None: sum = c.return_ascent(cur_state)
            else: sum += c.return_ascent(cur_state)

            if i != len(self.children) - 1: state.update_attrs_(cur_state)

        assert sum is not None
        if self.scalar is not None: sum += self.scalar
        if self.n_values > 1: sum /= self.n_values
        state.ascent = sum
        return self._update_params_or_step_with_next(state)

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
            self._set_child_(i, Chain(module))

    @torch.no_grad
    def step(self, state):
        if len(self.children) == 1:
            state.ascent = self.children[0].return_ascent(state)
            if self.scalar is not None: state.ascent *= self.scalar
            return self._update_params_or_step_with_next(state)

        prod = None
        for i, c in sorted(self.children.items(), key=lambda x: x[0]):
            if i == len(self.children) - 1: cur_state = state
            else: cur_state = state.copy(clone_ascent = True)

            if prod is None: prod = c.return_ascent(cur_state)
            else: prod *= c.return_ascent(cur_state)

            if i != len(self.children) - 1: state.update_attrs_(cur_state)

        assert prod is not None
        if self.scalar is not None: prod *= self.scalar
        state.ascent = prod
        return self._update_params_or_step_with_next(state)
