from collections.abc import Callable, Sequence
from typing import Any, overload

import torch
from typing_extensions import Concatenate, ParamSpec

from ...core import OptimizerModule
from .return_overrides import SetGrad

K = ParamSpec('K')

class Wrap(OptimizerModule):
    """
    Wraps any torch.optim.Optimizer.

    Sets .grad attribute to the current update and steps with the `optimizer`.

    Additionally, if this is not the last module, this takes the update of `optimizer`,
    undoes it and passes to the next module instead. That means you can chain multiple
    optimizers together.

    Args:
        optimizer (torch.optim.Optimizer): optimizer to wrap,
            or a callable (class) that constructs the optimizer.
        kwargs:
            if class is passed, kwargs are passed to the constructor.
            parameters are passed separately and automatically
            which is the point of passing a constructor
            instead of an optimizer directly.

    This can be constructed in two ways.
    .. code-block:: python
        wrapper = OptimizerWrapper(torch.optim.SGD(model.parameters(), lr = 0.1))
        # or
        wrapper = OptimizerWrapper(torch.optim.SGD, lr = 0.1)
    """

    @overload
    def __init__(self, optimizer: torch.optim.Optimizer): ...
    @overload
    # def __init__[**K](
    def __init__(
        self,
        optimizer: Callable[Concatenate[Any, K], torch.optim.Optimizer],
        *args: K.args,
        **kwargs: K.kwargs,
        # optimizer: abc.Callable[..., torch.optim.Optimizer],
        # *args,
        # **kwargs,
    ): ...
    def __init__(self, optimizer, *args, **kwargs):

        super().__init__({})
        self._optimizer_cls: torch.optim.Optimizer | Callable[..., torch.optim.Optimizer] = optimizer
        self._args = args
        self._kwargs = kwargs

    def _initialize_(self, params, set_passed_params):
        """Initializes this optimizer and all children with the given parameters."""
        super()._initialize_(params, set_passed_params=set_passed_params)
        if isinstance(self._optimizer_cls, torch.optim.Optimizer) or not callable(self._optimizer_cls):
            self.optimizer = self._optimizer_cls
        else:
            self.optimizer = self._optimizer_cls(params, *self._args, **self._kwargs)

    @torch.no_grad
    def step(self, state):
        # check attrs
        # if self.pass_closure:
        #     if state.closure is None: raise ValueError('ClosureOptimizerWrapper requires closure.')
        #     if state.ascent is not None:
        #         raise ValueError('pass_closure = True, means ascent must be None (not sure though)')

        params = self.get_params()

        if self.next_module is None:
            # set grad to ascent and make a step with the optimizer
            g = state.maybe_use_grad_(params)
            params.set_grad_(g)
            state.fx0 = self.optimizer.step()
            return state.get_loss()


        params_before_step = params.clone()

        g = state.maybe_use_grad_(params)
        params.set_grad_(g)
        state.fx0 = self.optimizer.step()

        # calculate update as difference in params
        state.ascent = params_before_step - params
        params.set_(params_before_step)
        return self.next_module.step(state)


class WrapClosure(OptimizerModule):
    """
    Wraps any torch.optim.Optimizer. This only works with modules with :code:`target = "Closure"` argument.
    The modified closure will be passed to the optimizer.

    Alternative any module can be turned into a closure module by using :any:`MakeClosure` module,
    in that case this should be placed after MakeClosure.

    Args:
        optimizer (torch.optim.Optimizer): optimizer to wrap,
            or a callable (class) that constructs the optimizer.
        kwargs:
            if class is passed, kwargs are passed to the constructor.
            parameters are passed separately and automatically
            which is the point of passing a constructor
            instead of an optimizer directly.

    This can be constructed in two ways.

    .. code-block:: python

        wrapper = OptimizerWrapper(torch.optim.SGD(model.parameters(), lr = 0.1))
        # or
        wrapper = OptimizerWrapper(torch.optim.SGD, lr = 0.1)

    """

    @overload
    def __init__(self, optimizer: torch.optim.Optimizer,): ...
    @overload
    def __init__(
        self,
        optimizer: Callable[Concatenate[Any, K], torch.optim.Optimizer],
        *args: K.args,
        **kwargs: K.kwargs,
        # optimizer: abc.Callable[..., torch.optim.Optimizer],
        # *args,
        # **kwargs,
    ): ...
    def __init__(self, optimizer, *args, **kwargs):

        super().__init__({})
        self._optimizer_cls: torch.optim.Optimizer | Callable[..., torch.optim.Optimizer] = optimizer
        self._args = args
        self._kwargs = kwargs

    def _initialize_(self, params, set_passed_params):
        """Initializes this optimizer and all children with the given parameters."""
        super()._initialize_(params, set_passed_params=set_passed_params)
        if isinstance(self._optimizer_cls, torch.optim.Optimizer) or not callable(self._optimizer_cls):
            self.optimizer = self._optimizer_cls
        else:
            self.optimizer = self._optimizer_cls(params, *self._args, **self._kwargs)

    @torch.no_grad
    def step(self, state):
        # check attrs
        # if self.pass_closure:
        #     if state.closure is None: raise ValueError('ClosureOptimizerWrapper requires closure.')
        #     if state.ascent is not None:
        #         raise ValueError('pass_closure = True, means ascent must be None (not sure though)')

        params = self.get_params()

        if self.next_module is None:
            # set grad to ascent and make a step with the optimizer
            state.fx0 = self.optimizer.step(state.closure) # type:ignore
            return state.get_loss()


        params_before_step = params.clone()
        state.fx0 = self.optimizer.step(state.closure) # type:ignore

        # calculate update as difference in params
        state.ascent = params_before_step - params
        params.set_(params_before_step)
        return self.next_module.step(state)

