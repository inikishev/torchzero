from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Literal

import torch

from ...core import (
    OptimizationVars,
    OptimizerModule,
    _ClosureType,
    _maybe_pass_backward,
    _ScalarLoss,
)
from ...tensorlist import TensorList


class GradientApproximatorBase(OptimizerModule, ABC):
    """Base gradient approximator class. This is an abstract class, please don't use it as the optimizer.

    When inheriting from this class the easiest way is to override `_make_ascent`, which should
    return the ascent direction (like approximated gradient).

    Args:
        defaults (dict[str, Any]): defaults
        requires_fx0 (bool):
            if True, makes sure to calculate fx0 beforehand.
            This means `_make_ascent` will always receive a pre-calculated `fx0` that won't be None.

        target (str, optional):
            determines what this module sets.

            "ascent" - it creates a new ascent direction but doesn't treat is as gradient.

            "grad" - it creates the gradient and sets it to `.grad` attributes (default).

            "closure" - it makes a new closure that sets the estimated gradient to the `.grad` attributes.
    """
    def __init__(self, defaults: dict[str, Any], requires_fx0: bool, target: Literal['ascent', 'grad', 'closure']):
        super().__init__(defaults, target)
        self.requires_fx0 = requires_fx0

    def _step_make_closure_(self, vars: OptimizationVars, params: TensorList):
        if vars.closure is None: raise ValueError("gradient approximation requires closure")
        closure = vars.closure

        if self.requires_fx0: fx0 = vars.evaluate_fx0_(False)
        else: fx0 = vars.fx0

        def new_closure(backward=True) -> _ScalarLoss:
            if backward:
                g, ret_fx0, ret_fx0_approx = self._make_ascent(closure, params, fx0)
                params.set_grad_(g)

                if ret_fx0 is None: return ret_fx0_approx # type:ignore
                return ret_fx0

            return closure(False)

        vars.closure = new_closure

    def _step_make_target_(self, vars: OptimizationVars, params: TensorList):
        if vars.closure is None: raise ValueError("gradient approximation requires closure")

        if self.requires_fx0: fx0 = vars.evaluate_fx0_(False)
        else: fx0 = vars.fx0

        g, vars.fx0, vars.fx0_approx = self._make_ascent(vars.closure, params, fx0)
        if self._default_step_target == 'ascent': vars.ascent = g
        elif self._default_step_target == 'grad': vars.set_grad_(g, params)
        else: raise ValueError(f"Unknown target {self._default_step_target}")

    @torch.no_grad
    def step(self, vars: OptimizationVars):
        params = self.get_params()
        if self._default_step_target == 'closure':
            self._step_make_closure_(vars, params)

        else:
            self._step_make_target_(vars, params)

        return self._update_params_or_step_with_next(vars, params)

    @abstractmethod
    @torch.no_grad
    def _make_ascent(
        self,
        # vars: OptimizationVars,
        closure: _ClosureType,
        params: TensorList,
        fx0: Any,
    ) -> tuple[TensorList, _ScalarLoss | None, _ScalarLoss | None]:
        """This should return a tuple of 3 elements:

        .. code:: py

            (ascent, fx0, fx0_approx)

        Args:
            closure (_ClosureType): closure
            params (TensorList): parameters
            fx0 (Any): fx0, can be None unless :target:`requires_fx0` is True on this module.

        Returns:
            (ascent, fx0, fx0_approx)
        """