from collections.abc import Callable, Iterable
from typing import cast
import torch

from ...tensorlist import TensorList

from ...core import OptimizerModule
from ..meta.chain import Chain


class Sum(OptimizerModule):
    def __init__(
        self,
        modules: Iterable[OptimizerModule | Iterable[OptimizerModule]],
    ):
        super().__init__({})
        modules = list(modules)
        for i,module in enumerate(modules):
            self._set_child_(i, Chain(module))

    @torch.no_grad
    def step(self, state):
        if len(self.children) == 1:
            state.ascent = self.children[0].return_ascent(state)
            return self._update_params_or_step_with_next(state)

        sum = None
        for i, c in sorted(self.children.items(), key=lambda x: x[0]):
            if i == len(self.children) - 1: cur_state = state
            else: cur_state = state.copy(clone_ascent = True)

            if sum is None: sum = c.return_ascent(cur_state)
            else: sum += c.return_ascent(cur_state)

            if i != len(self.children) - 1: state.update_attrs_(cur_state)

        assert sum is not None
        state.ascent = sum
        return self._update_params_or_step_with_next(state)

class Mean(OptimizerModule):
    def __init__(
        self,
        modules: Iterable[OptimizerModule | Iterable[OptimizerModule]],
    ):
        super().__init__({})
        modules = list(modules)
        for i,module in enumerate(modules):
            self._set_child_(i, Chain(module))

    @torch.no_grad
    def step(self, state):
        if len(self.children) == 1:
            state.ascent = self.children[0].return_ascent(state)
            return self._update_params_or_step_with_next(state)

        sum = None
        for i, c in sorted(self.children.items(), key=lambda x: x[0]):
            if i == len(self.children) - 1: cur_state = state
            else: cur_state = state.copy(clone_ascent = True)

            if sum is None: sum = c.return_ascent(cur_state)
            else: sum += c.return_ascent(cur_state)

            if i != len(self.children) - 1: state.update_attrs_(cur_state)

        assert sum is not None
        state.ascent = sum.div_(len(self.children))
        return self._update_params_or_step_with_next(state)


class Product(OptimizerModule):
    def __init__(
        self,
        modules: Iterable[OptimizerModule | Iterable[OptimizerModule]],
    ):
        super().__init__({})
        modules = list(modules)
        for i,module in enumerate(modules):
            self._set_child_(i, Chain(module))

    @torch.no_grad
    def step(self, state):
        if len(self.children) == 1:
            state.ascent = self.children[0].return_ascent(state)
            return self._update_params_or_step_with_next(state)

        prod = None
        for i, c in sorted(self.children.items(), key=lambda x: x[0]):
            if i == len(self.children) - 1: cur_state = state
            else: cur_state = state.copy(clone_ascent = True)

            if prod is None: prod = c.return_ascent(cur_state)
            else: prod *= c.return_ascent(cur_state)

            if i != len(self.children) - 1: state.update_attrs_(cur_state)

        assert prod is not None
        state.ascent = prod
        return self._update_params_or_step_with_next(state)


class Subtract(OptimizerModule):
    """a - b"""
    def __init__(
        self,
        a: OptimizerModule | Iterable[OptimizerModule],
        b: OptimizerModule | Iterable[OptimizerModule],
    ):
        super().__init__({})
        self._set_child_('a', Chain(a))
        self._set_child_('b', Chain(b))

    @torch.no_grad
    def step(self, state):
        state_copy = state.copy(clone_ascent = True)
        a = self.children['a'].return_ascent(state_copy)
        state.update_attrs_(state_copy)
        b = self.children['b'].return_ascent(state)

        state.ascent = a.sub_(b)
        return self._update_params_or_step_with_next(state)

class Divide(OptimizerModule):
    """numerator / denominator"""
    def __init__(
        self,
        numerator: OptimizerModule | Iterable[OptimizerModule],
        denominator: OptimizerModule | Iterable[OptimizerModule],
    ):
        super().__init__({})
        self._set_child_('numerator', Chain(numerator))
        self._set_child_('denominator', Chain(denominator))

    @torch.no_grad
    def step(self, state):
        state_copy = state.copy(clone_ascent = True)
        numerator = self.children['numerator'].return_ascent(state_copy)
        state.update_attrs_(state_copy)
        denominator = self.children['denominator'].return_ascent(state)

        state.ascent = numerator.div_(denominator)
        return self._update_params_or_step_with_next(state)


class Interpolate(OptimizerModule):
    """lerp. `out = self + weight * (tensors1 - self)`."""
    def __init__(
        self,
        input: OptimizerModule | Iterable[OptimizerModule],
        end: OptimizerModule | Iterable[OptimizerModule],
        weight: float,
    ):
        super().__init__({})
        self._set_child_('input', Chain(input))
        self._set_child_('end', Chain(end))
        self.weight = weight

    @torch.no_grad
    def step(self, state):
        state_copy = state.copy(clone_ascent = True)
        input = self.children['input'].return_ascent(state_copy)
        state.update_attrs_(state_copy)
        end = self.children['end'].return_ascent(state)

        state.ascent = input.lerp_(end, weight = self.weight)
        
        return self._update_params_or_step_with_next(state)

