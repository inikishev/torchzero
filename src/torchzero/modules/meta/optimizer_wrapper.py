import typing
from collections import abc

import torch

from ...core import OptimizerModule

class _Wrap(OptimizerModule):
    pass_closure:bool
    @typing.overload
    def __init__(self, optimizer: torch.optim.Optimizer): ...
    @typing.overload
    def __init__[**K](
        self,
        optimizer: abc.Callable[typing.Concatenate[typing.Any, K], torch.optim.Optimizer],
        *args: K.args,
        **kwargs: K.kwargs,
    ): ...
    def __init__(self, optimizer, *args, **kwargs):

        super().__init__({})
        self._optimizer_cls: torch.optim.Optimizer | abc.Callable[..., torch.optim.Optimizer] = optimizer
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
        if self.pass_closure:
            if state.closure is None: raise ValueError('ClosureOptimizerWrapper requires closure.')
            if state.ascent is not None:
                raise ValueError('pass_closure = True, means ascent must be None (not sure though)')

        params = None

        if self.next_module is None:
            if self.pass_closure:
                state.fx0_approx = self.optimizer.step(state.closure) # type:ignore
                return state.get_loss()

            params = self.get_params()

            # set grad to ascent and make a step with the optimizer
            # shouldn't accumulate here because params might already have autograd gradients
            if state.ascent is None: state.maybe_compute_grad_(params)
            else: params.set_grad_(state.ascent)
            state.fx0_approx = self.optimizer.step()
            return state.get_loss()

        else:
            if params is None: params = self.get_params()
            params_before_step = params.clone()

            if self.pass_closure: state.fx0_approx = self.optimizer.step(state.closure) # type:ignore
            else:
                if state.ascent is None: state.maybe_use_grad_(params)
                else: params.set_grad_(state.ascent)
                state.fx0_approx = self.optimizer.step()

            # calculate update as difference in params
            state.ascent = params_before_step - params
            params.set_(params_before_step)
            return self.next_module.step(state)


class Wrap(_Wrap):
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
    pass_closure:bool = False

class WrapClosure(_Wrap):
    """
    Wraps any torch.optim.Optimizer.

    Passes modified closure to the optimizer. Useful for making closure optimizers like LBFGS work with
    gradient transforms and gradient estimators.

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
    pass_closure:bool = True