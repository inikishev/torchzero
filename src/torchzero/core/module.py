from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch

from ..python_tools import ScalarType
from ..tensorlist import TensorList
from .tensorlist_optimizer import TensorListOptimizer, ParamsT


def _get_loss(fx0, fx0_approx):
    """Returns fx0 if it is not None otherwise fx0_approx"""
    if fx0 is None: return fx0_approx
    return fx0

ClosureType = Callable[..., ScalarType] #
"""
Closure example:
.. code-block:: python
    def closure(backward = True, **k):
        loss = model(inputs)
        if backward:
            optimizer.zero_grad()
            loss.backward(**k)
        return loss

This closure will also work with all built in pytorch optimizers including LBFGS, as well as and most custom ones.
"""

class OptimizationState:
    """Holds optimization state. This is usually automatically created by :any:`torchzero.optim.Modular`."""
    def __init__(self, closure: ClosureType | None, model: torch.nn.Module | None):

        self.closure: ClosureType | None = closure
        """A closure that reevaluates the model and returns the loss.
        The closure should accept `backward` boolean argument that is True by default, which,
        if True, sets `.grad` attributes of all learnable params, for example via `loss.backward()`.
        Closure can be None for some first order optimizers."""

        self.ascent: TensorList | None = None
        """Ascent direction, for example the gradients.
        Will be None on the first module in the chain.
        May remain none for modules that create a new closure."""

        self.fx0: ScalarType | None = None
        """Loss value strictly with initial parameters of the current step.
        If initial parameters have not been evaluated, this should be None."""

        self.fx0_approx: ScalarType | None = None
        """Loss value, could be sampled nearby the initial parameters.
        This is mainly used as the return value of the step method when fx0 is None."""

        self.grad: TensorList | None = None
        """Gradient if it has been computed, otherwise None."""

        self.model = model
        """Model (for higher order derivatives)"""

    def maybe_compute_grad_(self, params: TensorList) -> TensorList:
        """Computes gradient if it hasn't been computed already."""
        if self.grad is None:

            if self.closure is not None:
                with torch.enable_grad(): self.fx0 = self.closure(True) # pylint:disable = not-callable (???)
            self.grad = params.ensure_grad_().grad

        return self.grad

    def maybe_use_grad_(self, params: TensorList | None) -> TensorList:
        """If ascent direction is None, use cloned gradient as ascent direction.
        otherwise returns existing ascent direction.
        If gradient hasn't been computed, this also sets `fx0`."""
        if self.ascent is None:
            if params is None: raise ValueError()
            grad = self.maybe_compute_grad_(params)

            # clone the gradients to avoid modifying them.
            self.ascent = grad.clone()

        return self.ascent

    def get_loss(self):
        """Returns fx0 if it is not None otherwise fx0_approx"""
        if self.fx0 is None: return self.fx0_approx
        return self.fx0

    def copy(self, clone_ascent = False):
        """Copy this optimization state. This will not clone anything other than optionally ascent_direction.

        Args:
            clone_ascent (bool, optional): Whether to clone ascent direction. Defaults to False.

        Returns:
            A copy of this OptimizationState.
        """
        state = OptimizationState(self.closure, self.model)
        state.fx0 = self.fx0
        state.fx0_approx = self.fx0_approx
        state.grad = self.grad

        if clone_ascent and self.ascent is not None: state.ascent = self.ascent.clone()
        else: state.ascent = self.ascent

        return state

    def update_attrs_(self, state: "OptimizationState"):
        """Updates attributes of this state with attributes of another state."""
        if state.grad is not None: self.grad = state.grad
        if state.fx0 is not None: self.fx0 = state.fx0
        if state.fx0_approx is not None: self.fx0_approx = state.fx0_approx

class OptimizerModule(TensorListOptimizer, ABC):
    r"""Base class for all modules.

    Args:
        defaults (dict): dictionary with default parameters for the module.
        make_closure (bool, optional):
            if True, :any:`_update` method functions as a closure,
            otherwise it updates the ascent directly. Defaults to False.
    """
    def __init__(self, defaults: dict[str, Any], make_closure = False): # pylint:disable = super-init-not-called
        self._defaults = defaults
        self.next_module: OptimizerModule | None = None
        """next module that takes this module's state and continues working on it."""
        self.children: dict[Any, OptimizerModule] = {}
        """children modules."""
        self._initialized = False
        self._make_closure = make_closure

        self._has_custom_params = False
        """Signifies that :any:`self.set_params` was called on this to set custom params.
        When this is True, when parent calls :any:`_update_child_params_` with this module as child,
        nothing will happen, as this module already has parameters set."""

    def __repr__(self):
        if self._initialized: return super().__repr__()
        return f"uninitialized {self.__class__.__name__}()"

    def set_params(self, params: ParamsT):
        """
        Set parameters to this module. Use this to set per-parameter group settings.

        .. warning::
            (Warning from pytorch) Parameters need to be specified as collections that have a deterministic
            ordering that is consistent between runs. Examples of objects that don't
            satisfy those properties are sets and iterators over values of dictionaries.
        """
        self._initialize_(params)
        self._has_custom_params = True
        return self

    def _initialize_(self, params: ParamsT):
        """Initializes this optimizer and all children with the given parameters."""
        if isinstance(params, torch.Tensor): raise ValueError("Params must be an iterable of tensors, not torch.Tensor")
        params = list(params) # type:ignore
        # super().__init__, which is torch.optim.Optimizer.__init__,
        # calls self.add_param_group on each param group,
        # which in turn calls _update_child_params_,
        # which calls add_param_group on each child.
        super().__init__(params, self._defaults)
        self._initialized = True

    def _set_child_(self, name, child: "OptimizerModule"):
        """Set a child and initialize it's params."""
        self.children[name] = child
        if self._initialized:
            self._update_child_params_(child)

    def _set_next_module(self, next_module: "OptimizerModule"):
        """Set this module's child, overwriting existing children, and initialize it's params."""
        self.next_module = next_module
        if self._initialized:
            self._update_next_module_params_(next_module)

    def _update_next_module_params_(self, next_module: "OptimizerModule"):
        """Initializes or updates next module params with parameters of this module."""
        # Shouldn't forget that this method is overwritten by some modules
        # So if I update it I need to keep that in mind

        # if child is not initialized, torch.optim.Optimizer.__init__ is called on it by _initialize_ method
        if not next_module._initialized:
            next_module._initialize_(self._params)

        # otherwise to avoid calling __init__ multiple twice, we erase the param groups and readd them
        elif not next_module._has_custom_params:
            next_module.param_groups = []
            for group in self.param_groups:
                # it is important not to propagate all the settings
                # for example if this module has `lr` setting, and the child has a different `lr` setting,
                # we don't want to overwrite the child's `lr` setting
                next_module.add_param_group({"params": group["params"]})

    def _update_child_params_(self, child: "OptimizerModule"):
        """Initializes or updates child params with parameters of this module."""
        return self._update_next_module_params_(child)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        if self.next_module is not None: self._update_next_module_params_(self.next_module)
        for c in self.children.values():
            self._update_child_params_(c)

    def _update_params_or_step_with_next(self, state: OptimizationState, params: TensorList | None = None) -> ScalarType | None:
        """If this has no children, update params and return loss. Otherwise step with the child.

        Optionally pass params to not recreate them if you've already made them.

        Returns:
            Loss (fx0 or fx0_approx)
        """
        # if this has no children, update params and return loss.
        if self.next_module is None:
            if state.ascent is None: raise ValueError('Called _update_params_or_step_with_child but ascent_direction is None...')
            if params is None: params = self.get_params()
            params -= state.ascent
            return state.get_loss()

        # otherwise pass the updated ascent direction to the child
        return self.next_module.step(state)

    @torch.no_grad
    def _step_update_closure(self, state: OptimizationState) -> ScalarType | None:
        """Create a new closure which applies the `_update` method and pass it to the child."""
        if state.closure is None: raise ValueError('If `make_closure` is True, closure must be provided')

        params = self.get_params()
        closure = state.closure # closure shouldn't reference state attribute because it can be changed
        ascent_direction = state.ascent

        def update_closure(backward = True, **k):
            loss = closure(backward, **k)

            # on backward, update the ascent direction
            if backward:
                grad = self._update(state, ascent_direction) # type:ignore
                # set new ascent direction as gradients
                # (accumulation doesn't make sense here as closure always calls zero_grad)
                for p, g in zip(params,grad):
                    p.grad = g

            return loss

        # pass new closure to the child.
        if self.next_module is None:
            raise ValueError(f'{self.__class__.__name__} has no child to step with (maybe set make_closure to False?).')

        state.closure = update_closure
        return self.next_module.step(state)


    @torch.no_grad
    def _step_update_ascent_direction(self, state: OptimizationState) -> ScalarType | None:
        """Apply _update method to the ascent direction and pass it to the child, or make a step if child is None."""
        # the following code by default uses `_update` method which simple modules can override.
        # But you can also just override the entire `step`.

        params = None

        # cases where we would need params
        if state.ascent is None or self.next_module is None:
            params = self.get_params()

        # if this is the first module, it uses the gradients
        state.maybe_use_grad_(params)

        # apply the `_update` method
        ascent_direction = self._update(state, state.ascent) # type:ignore
        state.ascent = ascent_direction

        # peform an update with the ascent direction, or pass it to the child.
        return self._update_params_or_step_with_next(state, params=params)

    def return_ascent(self, state: OptimizationState, params=None) -> TensorList:
        if params is None: params = self.get_params()
        true_next = self.next_module
        self.next_module = _ReturnAscent(params) # type:ignore
        ascent: TensorList = self.step(state) # type:ignore
        self.next_module = true_next
        return ascent

    @torch.no_grad
    def step( # type:ignore # pylint:disable=signature-differs # pylint:disable = arguments-renamed
        self,
        state: OptimizationState
    ) -> ScalarType | None:
        """Perform a single optimization step to update parameter."""

        if self._make_closure: return self._step_update_closure(state)
        return self._step_update_ascent_direction(state)

    @torch.no_grad
    def _update(self, state: OptimizationState, ascent: TensorList) -> TensorList:
        """Update `ascent_direction` and return the new ascent direction (but it may update it in place).
        Make sure it doesn't return anything from `state` to avoid future modules modifying that in-place.

        Before calling `_update`, if ascent direction was not provided to `step`, it will be set to the gradients.

        After generating a new ascent direction with this `_update` method,
        if this module has no child, ascent direction will be subtracted from params.
        Otherwise everything is passed to the child."""
        raise NotImplementedError()

class _ReturnAscent:
    def __init__(self, params):
        self.params = params

    @torch.no_grad
    def step(self, state: OptimizationState) -> TensorList: # type:ignore
        update = state.maybe_use_grad_(self.params) # this will execute the closure which might be modified
        return update
