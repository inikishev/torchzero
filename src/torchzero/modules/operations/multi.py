from collections.abc import Iterable
import torch

from ...core import OptimizerModule

_Value = int | float | OptimizerModule | Iterable[OptimizerModule]

class Add(OptimizerModule):
    """add `value` to update. `value` can be a scalar, an OptimizerModule or sequence of OptimizerModules"""
    def __init__(self, value: _Value):
        super().__init__({})

        if not isinstance(value, (int, float)):
            self._set_child_('value', value)

        self.value = value

    @torch.no_grad()
    def _update(self, state, ascent):
        if isinstance(self.value, (int, float)):
            return ascent.add_(self.value)

        state_copy = state.copy(clone_ascent = True)
        v = self.children['value'].return_ascent(state_copy)
        return ascent.add_(v)


class Sub(OptimizerModule):
    """subtracts `value` from update. `value` can be a scalar, an OptimizerModule or sequence of OptimizerModules"""
    def __init__(self, subtrahend: _Value):
        super().__init__({})

        if not isinstance(subtrahend, (int, float)):
            self._set_child_('subtrahend', subtrahend)

        self.subtrahend = subtrahend

    @torch.no_grad()
    def _update(self, state, ascent):
        if isinstance(self.subtrahend, (int, float)):
            return ascent.sub_(self.subtrahend)

        state_copy = state.copy(clone_ascent = True)
        subtrahend = self.children['subtrahend'].return_ascent(state_copy)
        return ascent.sub_(subtrahend)

class RSub(OptimizerModule):
    """subtracts update from `value`. `value` can be a scalar, an OptimizerModule or sequence of OptimizerModules"""
    def __init__(self, minuend: _Value):
        super().__init__({})

        if not isinstance(minuend, (int, float)):
            self._set_child_('minuend', minuend)

        self.minuend = minuend

    @torch.no_grad()
    def _update(self, state, ascent):
        if isinstance(self.minuend, (int, float)):
            return ascent.sub_(self.minuend).neg_()

        state_copy = state.copy(clone_ascent = True)
        minuend = self.children['minuend'].return_ascent(state_copy)
        return ascent.sub_(minuend).neg_()

class Subtract(OptimizerModule):
    """Calculates `minuend - subtrahend`"""
    def __init__(
        self,
        minuend: OptimizerModule | Iterable[OptimizerModule],
        subtrahend: OptimizerModule | Iterable[OptimizerModule],
    ):
        super().__init__({})
        self._set_child_('minuend', minuend)
        self._set_child_('subtrahend', subtrahend)

    @torch.no_grad
    def step(self, state):
        state_copy = state.copy(clone_ascent = True)
        minuend = self.children['minuend'].return_ascent(state_copy)
        state.update_attrs_(state_copy)
        subtrahend = self.children['subtrahend'].return_ascent(state)

        state.ascent = minuend.sub_(subtrahend)
        return self._update_params_or_step_with_next(state)

class Mul(OptimizerModule):
    """multiplies update by `value`. `value` can be a scalar, an OptimizerModule or sequence of OptimizerModules"""
    def __init__(self, value: _Value):
        super().__init__({})

        if not isinstance(value, (int, float)):
            self._set_child_('value', value)

        self.value = value

    @torch.no_grad()
    def _update(self, state, ascent):
        if isinstance(self.value, (int, float)):
            return ascent.mul_(self.value)

        state_copy = state.copy(clone_ascent = True)
        v = self.children['value'].return_ascent(state_copy)
        return ascent.mul_(v)


class Div(OptimizerModule):
    """divides update by `value`. `value` can be a scalar, an OptimizerModule or sequence of OptimizerModules"""
    def __init__(self, denominator: _Value):
        super().__init__({})

        if not isinstance(denominator, (int, float)):
            self._set_child_('denominator', denominator)

        self.denominator = denominator

    @torch.no_grad()
    def _update(self, state, ascent):
        if isinstance(self.denominator, (int, float)):
            return ascent.div_(self.denominator)

        state_copy = state.copy(clone_ascent = True)
        denominator = self.children['denominator'].return_ascent(state_copy)
        return ascent.div_(denominator)

class RDiv(OptimizerModule):
    """`value` by update. `value` can be a scalar, an OptimizerModule or sequence of OptimizerModules"""
    def __init__(self, numerator: _Value):
        super().__init__({})

        if not isinstance(numerator, (int, float)):
            self._set_child_('numerator', numerator)

        self.numerator = numerator

    @torch.no_grad()
    def _update(self, state, ascent):
        if isinstance(self.numerator, (int, float)):
            return ascent.reciprocal_().mul_(self.numerator)

        state_copy = state.copy(clone_ascent = True)
        numerator = self.children['numerator'].return_ascent(state_copy)
        return ascent.reciprocal_().mul_(numerator)

class Divide(OptimizerModule):
    """calculates *numerator / denominator*"""
    def __init__(
        self,
        numerator: OptimizerModule | Iterable[OptimizerModule],
        denominator: OptimizerModule | Iterable[OptimizerModule],
    ):
        super().__init__({})
        self._set_child_('numerator', numerator)
        self._set_child_('denominator', denominator)

    @torch.no_grad
    def step(self, state):
        state_copy = state.copy(clone_ascent = True)
        numerator = self.children['numerator'].return_ascent(state_copy)
        state.update_attrs_(state_copy)
        denominator = self.children['denominator'].return_ascent(state)

        state.ascent = numerator.div_(denominator)
        return self._update_params_or_step_with_next(state)


class Pow(OptimizerModule):
    """takes ascent to the power of `value`. `value` can be a scalar, an OptimizerModule or sequence of OptimizerModules"""
    def __init__(self, power: _Value):
        super().__init__({})

        if not isinstance(power, (int, float)):
            self._set_child_('power', power)

        self.power = power

    @torch.no_grad()
    def _update(self, state, ascent):
        if isinstance(self.power, (int, float)):
            return ascent.pow_(self.power)

        state_copy = state.copy(clone_ascent = True)
        power = self.children['power'].return_ascent(state_copy)
        return ascent.pow_(power)

class RPow(OptimizerModule):
    """takes `value` to the power of ascent. `value` can be a scalar, an OptimizerModule or sequence of OptimizerModules"""
    def __init__(self, base: _Value):
        super().__init__({})

        if not isinstance(base, (int, float)):
            self._set_child_('base', base)

        self.base = base

    @torch.no_grad()
    def _update(self, state, ascent):
        if isinstance(self.base, (int, float)):
            return self.base ** ascent

        state_copy = state.copy(clone_ascent = True)
        base = self.children['base'].return_ascent(state_copy)
        return base.pow_(ascent)

class Power(OptimizerModule):
    """calculates *base ^ power*"""
    def __init__(
        self,
        base: OptimizerModule | Iterable[OptimizerModule],
        power: OptimizerModule | Iterable[OptimizerModule],
    ):
        super().__init__({})
        self._set_child_('base', base)
        self._set_child_('power', power)

    @torch.no_grad
    def step(self, state):
        state_copy = state.copy(clone_ascent = True)
        base = self.children['base'].return_ascent(state_copy)
        state.update_attrs_(state_copy)
        power = self.children['power'].return_ascent(state)

        state.ascent = base.pow_(power)
        return self._update_params_or_step_with_next(state)


class Lerp(OptimizerModule):
    """Linear interpolation between update and `end` based on scalar `weight`.

    `out = update + weight * (end - update)`"""
    def __init__(self, end: OptimizerModule | Iterable[OptimizerModule], weight: float):
        super().__init__({})

        self._set_child_('end', end)
        self.weight = weight

    @torch.no_grad()
    def _update(self, state, ascent):

        state_copy = state.copy(clone_ascent = True)
        end = self.children['end'].return_ascent(state_copy)
        return ascent.lerp_(end, self.weight)


class Interpolate(OptimizerModule):
    """Does a linear interpolation of two module's updates - `start` (given by input), and `end`, based on a scalar
    `weight`.

    `out = input + weight * (end - input)`"""
    def __init__(
        self,
        input: OptimizerModule | Iterable[OptimizerModule],
        end: OptimizerModule | Iterable[OptimizerModule],
        weight: float,
    ):
        super().__init__({})
        self._set_child_('input', input)
        self._set_child_('end', end)
        self.weight = weight

    @torch.no_grad
    def step(self, state):
        state_copy = state.copy(clone_ascent = True)
        input = self.children['input'].return_ascent(state_copy)
        state.update_attrs_(state_copy)
        end = self.children['end'].return_ascent(state)

        state.ascent = input.lerp_(end, weight = self.weight)

        return self._update_params_or_step_with_next(state)

class AddMagnitude(OptimizerModule):
    """Add `value` multiplied by sign of the ascent, i.e. this adds `value` to the magnitude of the update.

    Args:
        value (Value): value to add to magnitude, either a float or an OptimizerModule.
        add_to_zero (bool, optional):
            if True, adds `value` to 0s. Otherwise, zeros remain zero.
            Only has effect if value is a float. Defaults to True.
    """
    def __init__(self, value: _Value, add_to_zero=True):
        super().__init__({})

        if not isinstance(value, (int, float)):
            self._set_child_('value', value)

        self.value = value
        self.add_to_zero = add_to_zero

    @torch.no_grad()
    def _update(self, state, ascent):
        if isinstance(self.value, (int, float)):
            if self.add_to_zero: return ascent.add_(ascent.clamp_magnitude(min=1).sign_().mul_(self.value))
            return ascent.add_(ascent.sign_().mul_(self.value))

        state_copy = state.copy(clone_ascent = True)
        v = self.children['value'].return_ascent(state_copy)
        return ascent.add_(v.abs_().mul_(ascent.sign()))