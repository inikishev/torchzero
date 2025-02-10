import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal
from typing_extensions import Self, TypeAlias

import torch
from torch.optim.optimizer import ParamsT

from ..tensorlist import TensorList
from ..utils.python_tools import _ScalarLoss, flatten

from .tensorlist_optimizer import (
    TensorListOptimizer,
    _ClosureType,
    _maybe_pass_backward,
)

def _get_loss(fx0, fx0_approx):
    """Returns fx0 if it is not None otherwise fx0_approx"""
    if fx0 is None: return fx0_approx
    return fx0


class OptimizationVars:
    """Holds optimization variables. This is usually automatically created by :any:`torchzero.optim.Modular`."""
    def __init__(self, closure: _ClosureType | None, model: torch.nn.Module | None):

        self.closure: _ClosureType | None = closure
        """A closure that reevaluates the model and returns the loss.
        The closure should accept `backward` boolean argument that is True by default, which,
        if True, sets `.grad` attributes of all learnable params, for example via `loss.backward()`.
        Closure can be None for most first order optimizers."""

        self.ascent: TensorList | None = None
        """Ascent direction, for example the gradients.
        Will be None on the first module in the chain.
        May remain none for modules that create a new closure."""

        self.fx0: _ScalarLoss | None = None
        """Loss value strictly with initial parameters of the current step.
        If initial parameters have not been evaluated, this should be None."""

        self.fx0_approx: _ScalarLoss | None = None
        """Loss value, could be sampled nearby the initial parameters.
        This is mainly used as the return value of the step method when fx0 is None."""

        self.grad: TensorList | None = None
        """Gradient if it has been computed, otherwise None.

        Gradient must be evaluated strictly with initial parameters of the current step"""

        self.model = model
        """model itself (torch.nn.Module) if it was passed, otherwise None."""

        self.post_step_hooks = []
        """callables that get executed after each step. Used by periodic SWA to reset momentum when setting model parameters to SWA.

        Signature:

        .. code:: py

            def hook(optimizer: ModularOptimizer, state: OptimizationState) -> None:
                ...
        """

    def maybe_compute_grad_(self, params: TensorList | None) -> TensorList:
        """Computes gradient if it hasn't been computed already, and returns it"""
        if self.grad is None:
            if params is None: raise ValueError()
            if self.closure is not None:
                with torch.enable_grad(): self.fx0 = self.closure() # pylint:disable = not-callable (???)
            self.grad = params.ensure_grad_().grad

        return self.grad

    def maybe_use_grad_(self, params: TensorList | None) -> TensorList:
        """If ascent direction is None, use cloned gradient as ascent direction and returns it.
        Otherwise does nothing and returns existing ascent direction.
        If gradient hasn't been computed, this also sets `fx0`."""
        if self.ascent is None:
            self.ascent = self.maybe_compute_grad_(params).clone()

        return self.ascent

    def set_grad_(self, grad: TensorList, params: TensorList):
        """Sets gradient to this state and to params"""
        self.grad = grad
        params.set_grad_(grad)

    def evaluate_fx0_(self, backward=True) -> _ScalarLoss:
        """if fx0 is None or if backward is True and self.grad is None, evaluates closure and sets them. Returns fx0"""
        if self.fx0 is not None:
            if backward and self.grad is None:
                warnings.warn('evaluating fx0 with backward=True after it has already been evaluated with backward=False. Something may be inefficient.')
                with torch.enable_grad(): self.closure() # set grad #type:ignore
            return self.fx0

        if self.closure is None: raise ValueError("Closure is None")
        loss = self.fx0 = _maybe_pass_backward(self.closure, backward)
        return loss

    def evaluate_fx0_approx_(self, backward=True) -> _ScalarLoss:
        """evaluates closure, sets self.fx0_approx and returns it"""
        if self.closure is None: raise ValueError("Closure is None")
        loss = self.fx0_approx = _maybe_pass_backward(self.closure, backward)
        return loss

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
        vars = OptimizationVars(self.closure, self.model)
        vars.fx0 = self.fx0
        vars.fx0_approx = self.fx0_approx
        vars.grad = self.grad

        if clone_ascent and self.ascent is not None: vars.ascent = self.ascent.clone()
        else: vars.ascent = self.ascent

        return vars

    def update_attrs_(self, vars: "OptimizationVars"):
        """Updates attributes of this state with attributes of another state.

        This updates `grad`, `fx0` and `fx0_approx`."""
        if vars.grad is not None: self.grad = vars.grad
        if vars.fx0 is not None: self.fx0 = vars.fx0
        if vars.fx0_approx is not None: self.fx0_approx = vars.fx0_approx


    def add_post_step_hook(self, hook: Callable):
        """add a hook that runs after each step. The hook should look like this:
        .. code:: py
            def hook(optimizer: tz.optim.Modular, state: tz.core.OptimizationState): ...
        """
        self.post_step_hooks.append(hook)

_Targets = Literal['ascent', 'grad', 'closure',]
class OptimizerModule(TensorListOptimizer, ABC): # type:ignore
    r"""Base class for all modules.

    Args:
        defaults (dict): dictionary with default parameters for the module.
        target (str, optional):
            determines how _update method is used in the default step method.

            "ascent" - it updates the ascent

            "grad" - it updates the gradient (and sets `.grad` attributes to updated gradient).

            "closure" - it makes a new closure that sets the updated ascent to the .`grad` attributes.
    """
    IS_LR_MODULE = False
    def __init__(self, defaults: dict[str, Any], target: Literal['ascent', 'grad', 'closure',] = 'ascent'): # pylint:disable = super-init-not-called
        # there can only be 1 LR module, which is placed in the appropriate location among other modules.
        # scheduling and per-parameter "lr" options will be routed to that module.
        # otherwise, since many update rules like Adam have baked in lr, if multiple such modules are used,
        # any lr modification gets applied multiple times.
        # Some optimizers will automatically be fused if followed an LR() module (only LR specifically is supported).
        if not self.IS_LR_MODULE:
            if 'lr' in defaults:
                warnings.warn(
                    f'{self.__class__.__name__} got an "lr" default, but it is not an LR module.\
                    To support lr scheduling and per-parameter options, rename "lr" to "alpha" and set the default value to 1.\
                    If this is a learning rate module, set a class attribute `IS_LR_MODULE=True`.'
                )

        self._defaults = defaults
        self.next_module: OptimizerModule | None = None
        """next module that takes this module's state and continues working on it."""
        self.children: dict[Any, OptimizerModule] = {}
        """children modules."""
        self._initialized = False
        """True if torch.optim.Optimzer.__init__ was called on this meaning this optimizer has parameters."""
        self._default_step_target: Literal['ascent', 'grad', 'closure'] = target
        """'ascent', 'grad' or 'closure'"""

        self._has_custom_params = False
        """Signifies that :any:`self.set_params` was called on this to set custom params.
        When this is True, when parent calls :any:`_update_child_params_` with this module as child,
        nothing will happen, as this module already has parameters set."""

        self._passed_params: list[torch.Tensor] | list[dict[str, Any]] | None = None
        """list of parameters or parameter groups that were passed to this module and will get passed to child modules."""

        self.post_init_hooks: list[Callable[[Any, Self], Any]] = []
        """Hooks that run once after a ModularOptimizer is initialized with this module.

        Signature:

        .. code:: py

            def hook(optimizer: ModularOptimizer, module: OptimizerModule) -> None:
                ...

        where `module` is this module.
        """

    def __repr__(self):
        if self._initialized: return super().__repr__()
        return f"uninitialized {self.__class__.__name__}()"

    def set_params(self, params: ParamsT):
        """
        Set parameters to this module. Use this to set per-parameter group settings.
        """
        self._initialize_(params, set_passed_params = False)
        self._has_custom_params = True
        return self

    def _initialize_(self, params: ParamsT, set_passed_params: bool):
        """Initializes this optimizer and all children with the given parameters."""
        if isinstance(params, torch.Tensor): raise ValueError("Params must be an iterable of tensors, not torch.Tensor")
        params_list = list(params)
        if set_passed_params: self._passed_params = params_list.copy() # type:ignore

        # super().__init__, which is torch.optim.Optimizer.__init__,
        # calls self.add_param_group on each param group,
        # which in turn calls _update_child_params_,
        # which calls add_param_group on each child.
        super().__init__(params_list.copy(), self._defaults) # type:ignore
        self._initialized = True

    def _set_child_(self, name, child: "_Chainable"):
        """Set a child and initialize it's params."""
        if not isinstance(child, OptimizerModule): child = _Chain(child)
        self.children[name] = child
        if self._initialized:
            self._update_child_params_(child)

    def _update_child_params_(self, child: "OptimizerModule"):
        """Initializes or updates child params with parameters of this module."""
        return self._update_next_module_params_(child)

    def _set_next_module(self, next_module: "OptimizerModule"):
        """Set next module and initialize it's params."""
        self.next_module = next_module
        if self._initialized:
            self._update_next_module_params_(next_module)

    def _update_next_module_params_(self, next_module: "OptimizerModule"):
        """Initializes or updates next module params with parameters of this module."""
        # Shouldn't forget that this method is overwritten by some modules
        # So if I update it I need to keep that in mind
        if self._passed_params is None:
            raise RuntimeError(
                f"{self.__class__.__name__} is not initialized, but _update_next_module_params_\
                was called with next_module = {next_module.__class__.__name__}"
            )

        # if child is not initialized, torch.optim.Optimizer.__init__ is called on it by _initialize_ method
        if not next_module._initialized:
            next_module._initialize_(self._passed_params, set_passed_params=True)

        # otherwise to avoid calling __init__ multiple twice, we erase the param groups and readd them
        elif not next_module._has_custom_params:
            next_module.param_groups = []
            for group in self._passed_params:
                if isinstance(group, torch.Tensor): group = {"params": group}
                next_module.add_param_group(group)

        else:
            # still pass per-parameter settings so that they propagate to further modules
            next_module._passed_params = self._passed_params.copy()


    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)

        if self.next_module is not None: self._update_next_module_params_(self.next_module)
        for c in self.children.values():
            self._update_child_params_(c)

    def _update_params_or_step_with_next(self, vars: OptimizationVars, params: TensorList | None = None) -> _ScalarLoss | None:
        """If this has no children, update params and return loss. Otherwise step with the next module.

        Optionally pass params to not recreate them if you've already made them.

        Returns:
            Loss (fx0 or fx0_approx)
        """
        # if this has no children, update params and return loss.
        if self.next_module is None:
            if vars.ascent is None: raise ValueError('Called _update_params_or_step_with_child but ascent_direction is None...')
            if params is None: params = self.get_params()
            params -= vars.ascent # type:ignore
            return vars.get_loss()

        # otherwise pass the updated ascent direction to the child
        return self.next_module.step(vars)

    @torch.no_grad
    def _step_update_closure(self, vars: OptimizationVars) -> _ScalarLoss | None:
        """Create a new closure which applies the `_update` method and passes it to the next module."""
        if vars.closure is None: raise ValueError('If target == "closure", closure must be provided')

        params = self.get_params()
        closure = vars.closure # closure shouldn't reference state attribute because it can be changed
        ascent_direction = vars.ascent

        def update_closure(backward = True):
            loss = _maybe_pass_backward(closure, backward)

            # on backward, update the ascent direction
            if backward:
                grad = self._update(vars, ascent_direction) # type:ignore
                # set new ascent direction as gradients
                # (accumulation doesn't make sense here as closure always calls zero_grad)
                for p, g in zip(params,grad):
                    p.grad = g

            return loss

        # pass new closure to the child.
        # if self.next_module is None:
        #     raise ValueError(f'{self.__class__.__name__} has no child to step with (maybe set "target" from "closure" to something else??).')

        vars.closure = update_closure
        return self._update_params_or_step_with_next(vars)


    @torch.no_grad
    def _step_update_target(self, vars: OptimizationVars) -> _ScalarLoss | None:
        """Apply _update method to the ascent direction and pass it to the child, or make a step if child is None."""
        # the following code by default uses `_update` method which simple modules can override.
        # But you can also just override the entire `step`.

        params = None

        # update ascent direction
        if self._default_step_target == 'ascent':
            # if this is the first module, it uses the gradients
            if vars.grad is None: params = self.get_params()
            t = vars.maybe_use_grad_(params)
            vars.ascent = self._update(vars, t)

        # update gradients
        elif self._default_step_target == 'grad':
            if params is None: params = self.get_params()
            g = vars.maybe_compute_grad_(params)
            g = self._update(vars, g)
            vars.set_grad_(g, params)
        else:
            raise ValueError(f"Invalid {self._default_step_target = }")

        # peform an update with the new state, or pass it to the child.
        return self._update_params_or_step_with_next(vars, params=params)

    @torch.no_grad
    def step( # type:ignore # pylint:disable=signature-differs # pylint:disable = arguments-renamed
        self,
        vars: OptimizationVars
    ) -> _ScalarLoss | None:
        """Perform a single optimization step to update parameter."""

        if self._default_step_target == 'closure': return self._step_update_closure(vars)
        return self._step_update_target(vars)

    @torch.no_grad
    def _update(self, vars: OptimizationVars, ascent: TensorList) -> TensorList:
        """Update `ascent_direction` and return the new ascent direction (but it may update it in place).
        Make sure it doesn't return anything from `self.state` to avoid future modules modifying that in-place.

        Before calling `_update`, if ascent direction was not provided to `step`, it will be set to the gradients.

        After generating a new ascent direction with this `_update` method,
        if this module has no child, ascent direction will be subtracted from params.
        Otherwise everything is passed to the child."""
        params = self.get_params()
        gradients = ascent.grad
        if gradients is None: gradients = [None] * len(params)
        settings = tuple(self.get_all_group_keys(list).items())

        new_ascent = TensorList()
        for i, (asc, param, grad) in enumerate(zip(ascent, params, gradients)):
            kwargs = {"vars": vars, "ascent": asc, "param": param, "grad": grad}
            kwargs.update({k:v[i] for k,v in settings})
            new_ascent.append(self._single_tensor_update(**kwargs))
        return new_ascent


    def _single_tensor_update(self, vars: OptimizationVars, ascent: torch.Tensor, param: torch.Tensor, grad: torch.Tensor | None, **kwargs) -> torch.Tensor:
        """Update function for a single tensor.

        Args:
            vars (OptimizationState): holds loss, gradients, etc.
            ascent (torch.Tensor): update tensor.
            param (torch.Tensor): parameter tensor.
            grad (torch.Tensor | None): gradient tensor (may be None)
            **kwargs: all per-parameter settings (stuff that you put in `defaults = dict(beta1=beta1, beta2=beta2, eps=eps)`).
        """
        raise NotImplementedError()

    def return_ascent(self, vars: OptimizationVars, params=None) -> TensorList:
        """step with this module and return the ascent as tensorlist"""
        if params is None: params = self.get_params()
        true_next = self.next_module
        self.next_module = _ReturnAscent(params) # type:ignore
        ascent: TensorList = self.step(vars) # type:ignore
        self.next_module = true_next
        return ascent

    def reset_stats(self):
        """Resets running stats of this optimizer such as momentum. This is meant to be used stop all
        momentum when significantly changing model parameters, for example when setting model parameters
        to weighted average every once in a while, like periodic SWA does. Pediodic resetting
        may also be beneficial for some optimizers.
        By default this method completely clears per-parameter state.
        Modules may override this to provide different functionality."""
        for g in self.param_groups:
            for p in g['params']:
                state = self.state[p]
                for k in state.copy().keys(): del state[k]


class _ReturnAscent:
    __slots__ = ('IS_LR_MODULE', 'params', 'children', 'next_module', )
    def __init__(self, params):
        self.params = params
        self.IS_LR_MODULE = False

        self.children = {}
        self.next_module = None

    @torch.no_grad
    def step(self, vars: OptimizationVars) -> TensorList: # type:ignore
        update = vars.maybe_use_grad_(self.params) # this will execute the closure which might be modified
        return update


class _MaybeReturnAscent(OptimizerModule):
    """utility module that either returns ascent or updates the parameters depending on `_return_ascent`, used in Chain."""
    def __init__(self):
        super().__init__({})
        self._return_ascent = False

    @torch.no_grad
    def step(self, vars: OptimizationVars):
        assert self.next_module is None, self.next_module

        if self._return_ascent:
            return vars.ascent

        return self._update_params_or_step_with_next(vars)

_Chainable = OptimizerModule | Iterable[OptimizerModule]

class _Chain(OptimizerModule):
    """
    Utility module that chains multiple modules together, usually used by other modules.
    """
    def __init__(self, *modules: _Chainable):
        super().__init__({})
        flat_modules: list[OptimizerModule] = flatten(modules)

        if any(not hasattr(i, "step") for i in flat_modules):
            raise TypeError(f"One of the modules is not an OptimizerModule, got {[i.__class__.__name__ for i in flat_modules]}")

        # first module is chain's child, second module is first module's child, etc
        self._set_child_('first', flat_modules[0])
        if len(flat_modules) > 1:
            for i, m in enumerate(flat_modules[:-1]):
                m._set_next_module(flat_modules[i+1])

            self._last_module = flat_modules[-1]

        self._chain_modules = flat_modules

    @torch.no_grad
    def step(self, vars: OptimizationVars):
        # no next module, step with the child
        if self.next_module is None:
            return self.children['first'].step(vars)

        # return ascent and pass it to next module
        # we do this because updating parameters directly is often more efficient
        params = self.get_params()
        self._last_module.next_module = _ReturnAscent(params) # type:ignore
        vars.ascent: TensorList = self.children['first'].step(vars) # type:ignore
        self._last_module.next_module = None

        return self._update_params_or_step_with_next(vars)