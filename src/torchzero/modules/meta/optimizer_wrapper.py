import typing as T
from collections import abc

import torch

from ...core import OptimizerModule, _get_loss

class OptimizerWrapper(OptimizerModule):
    def __init__(self, optimizer: torch.optim.Optimizer):
        """
        Wraps any torch.optim.Optimizer.
        If this is the last module, sets ascent direction as .grad attribute and performs a step with the optimizer.
        Otherwise uses minus update of the optimizer as new ascent direction.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to wrap.
        """

        super().__init__({})
        self.optimizer = optimizer

    @torch.no_grad
    def step(self, state):
        if state.ascent_direction is None: raise ValueError

        params = self.get_params()
        params.accumulate_grad_(state.ascent_direction)

        if self.child is None:
            state.fx0_approx = self.optimizer.step()
            return state.get_loss()

        else:
            params_before_step = params.clone()
            state.fx0_approx = self.optimizer.step()
            state.ascent_direction = params_before_step - params
            params.set_(params_before_step)
            return self.child.step(state)

class ClosureOptimizerWrapper(OptimizerModule):
    def __init__(self, optimizer: torch.optim.Optimizer):
        """
        Wraps any torch.optim.Optimizer.
        If this is the last module, performs a step with the optimizer passing it the closure.
        Otherwise uses minus update of the optimizer as new ascent direction.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to wrap.
        """

        super().__init__({})
        self.optimizer = optimizer

    @torch.no_grad
    def step(self, state):
        if state.closure is None: raise ValueError('ClosureOptimizerWrapper requires closure.')
        if state.ascent_direction is not None: raise ValueError('ascent_direction must be None (maybe not???)')

        if self.child is None:
            state.fx0_approx = self.optimizer.step(state.closure) # type:ignore
            return state.get_loss()

        else:
            params = self.get_params()
            params_before_step = params.clone()
            state.fx0_approx = self.optimizer.step(state.closure) # type:ignore
            state.ascent_direction = params_before_step - params
            params.set_(params_before_step)
            return self.child.step(state)

class UninitializedClosureOptimizerWrapper(OptimizerModule):
    def __init__[**K](
        self,
        optimizer_cls: abc.Callable[T.Concatenate[T.Any, K], torch.optim.Optimizer],
        /,
        *args: K.args,
        **kwargs: K.kwargs,
    ):
        """
        Wraps any torch.optim.Optimizer.
        If this is the last module, performs a step with the optimizer passing it the closure.
        Otherwise uses minus update of the optimizer as new ascent direction.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to wrap.
        """

        super().__init__({})
        self._optimizer_cls = optimizer_cls
        self._args = args
        self._kwargs = kwargs

    def _initialize_(self, params):
        """Initializes this optimizer and all children with the given parameters."""
        super()._initialize_(params)
        self.optimizer = self._optimizer_cls(params, *self._args, **self._kwargs)

    @torch.no_grad
    def step(self, state):
        if state.closure is None: raise ValueError('ClosureOptimizerWrapper requires closure.')
        if state.ascent_direction is not None: raise ValueError('ascent_direction must be None (maybe not???)')

        if self.child is None:
            state.fx0_approx = self.optimizer.step(state.closure) # type:ignore
            return state.get_loss()

        else:
            params = self.get_params()
            params_before_step = params.clone()
            state.fx0_approx = self.optimizer.step(state.closure) # type:ignore
            state.ascent_direction = params_before_step - params
            params.set_(params_before_step)
            return self.child.step(state)

# TODO: optimizer whatevering