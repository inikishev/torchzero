import warnings
from abc import ABC, abstractmethod
from collections import ChainMap, defaultdict
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from typing import Any, overload

import torch

from ..utils import (
    Init,
    ListLike,
    Params,
    _make_param_groups,
    get_state_vals,
)
from ..utils.python_tools import flatten


# region Vars
# ----------------------------------- vars ----------------------------------- #
class Vars:
    def __init__(
        self,
        params: list[torch.Tensor],
        closure: Callable | None,
        model: torch.nn.Module | None,
        current_step: int,
    ):
        self.params: list[torch.Tensor] = params
        """List of all parameters with requires_grad = True."""

        self.closure = closure
        """A closure that reevaluates the model and returns the loss, None if it wasn't specified"""

        self.model = model
        """torch.nn.Module object of the model, None if it wasn't specified."""

        self.current_step: int = current_step
        """global current step, starts at 0"""

        self.update: list[torch.Tensor] | None = None
        """
        current update, at the end this is subtracted from model parameters unless it is None.

        If closure is None, this is initially set to cloned gradient. Otherwise this is set to None.
        """

        self.grad: list[torch.Tensor] | None = None
        """gradient at current point. If closure is not None, this is set to None and can be calculated if needed."""

        self.loss: torch.Tensor | float | None = None
        """loss at current point."""

        self.loss_approx: torch.Tensor | float | None = None
        """loss at a point near current point. This can be useful as some modules only calculate loss at perturbed points,
        whereas some other modules require loss strictly at current point."""

        self.post_step_hooks: list[Callable[[Modular, Vars]]] = []
        """list of functions to be called after optimizer step.
        The signature is:

        .. code:: py

            def hook(optimizer: Modular, vars: Vars): ...

        """

        self.is_last: bool = False
        """
        This is set to True if current module is last or next to last before a learning rate module.
        If learning rate module is next, :code:`vars.last_module_lrs` will be set to a list of per-parameter learning rates,
        otherwise it will be None and learning rate is assumed to be 1.

        This can be used to apply the update directly to parameters instead of calculating a new update.
        If you apply the update manually, make sure to scale the update by :code:`vars.last_module_lrs` if it is not None,
        and set :code:`vars.update = None` and :code:`vars.stop = True` to prevent learning rate module from stepping
        and update from being applied twice.

        If current module has children, :code:`is_last` will always be False, and :code:`nested_is_last`
        is used instead. Note that :code:`nested_is_last` requires more careful handling because
        if it is True, children will also receive vars with :code:`nested_is_last = True`. If applying
        the update directly, make sure the module is not a child by checking :code:`if not self.is_child`.
        """

        self.nested_is_last: bool = False
        """
        This is set to True if current module is last or next to last before a learning rate module, and
        current module either has children or is a child. Please refer to :code:`vars.is_last` documentation
        for more details.
        """
        self.last_module_lrs: list[float] | None = None
        """
        This is set to a list of per-parameter learning rates if current module is next to last before a
        learning rate module, otherwise this is set to None. Ignore this unless you are manually applying
        update to parameters.
        """

        self.stop: bool = False
        """if True, all following modules will be skipped."""

    def get_loss(self, backward: bool) -> torch.Tensor | float:
        """evaluates loss is it hasn't been evaluated yet and returns it. This should only be called at current point."""
        if self.loss is None:
            if self.closure is None: raise RuntimeError("closure is None")
            if backward:
                with torch.enable_grad():
                    self.loss = self.loss_approx = self.closure()

                # initializing to zeros_like is equivalent to using zero_grad with set_to_none = False.
                # it is technically a more correct approach for when some parameters conditionally receive gradients
                # and in this case it shouldn't be slower.
                self.grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
            else:
                self.loss = self.loss_approx = self.closure(False)

        # if self.loss was not None, above branch wasn't executed because loss has already been evaluated, but without backward since self.grad is None.
        # and now it is requested to be evaluated with backward.
        if backward and self.grad is None:
            warnings.warn('get_loss was called with backward=False, and then with backward=True so it had to be re-evaluated, so the closure was evaluated twice where it could have been evaluated once.')
            if self.closure is None: raise RuntimeError("closure is None")

            with torch.enable_grad():
                self.loss = self.loss_approx = self.closure()

            self.grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        return self.loss # type:ignore

    def get_grad(self) -> list[torch.Tensor]:
        """evaluates grad if it hasn't been evaluated yet and returns it. This should only be called at current point."""
        if self.grad is None:
            if self.closure is None: raise RuntimeError("closure is None")
            self.get_loss(backward=True) # evaluate and set self.loss and self.grad

        assert self.grad is not None
        return self.grad

    def get_update(self) -> list[torch.Tensor]:
        """returns update, if it hasn't been assigned, sets it to cloned gradient, which is calculated if needed. This should only be called at current point."""
        if self.update is None: self.update = [g.clone() for g in self.get_grad()]
        return self.update

    def clone(self, clone_update: bool):
        """clone vars, optionally clone update. Make sure to clone vars when passing it to a child."""
        copy = Vars(params = self.params, closure=self.closure, model=self.model, current_step=self.current_step)

        if clone_update and self.update is not None:
            copy.update = [u.clone() for u in self.update]
        else:
            copy.update = self.update

        copy.grad = self.grad
        copy.loss = self.loss
        copy.loss_approx = self.loss_approx
        copy.post_step_hooks = self.post_step_hooks

        return copy

    def update_attrs_from_clone_(self, vars: "Vars"):
        """when Vars is copied, a child might evaluate loss and gradients, this updates those from it to the main Vars.
        This updates loss, loss_approx, grad"""
        if self.loss is None: self.loss = vars.loss
        if self.loss_approx is None: self.loss_approx = vars.loss_approx
        if self.grad is None: self.grad = vars.grad
# endregion

# region Module
# ---------------------------------- module ---------------------------------- #
class Module(ABC):
    def __init__(self, defaults: dict[str, Any] | None = None):
        if defaults is None: defaults = {}
        self.defaults: dict[str, Any] = defaults

        # settings are stored like state in per-tensor defaultdict, with per-parameter overrides possible
        # 0 - this module specific per-parameter setting overrides set via `set_param_groups` - highest priority
        # 1 - global per-parameter setting overrides in param_groups passed to Modular - medium priority
        # 2 - `defaults` - lowest priority
        self.settings: defaultdict[torch.Tensor, ChainMap[str, Any]] = defaultdict(lambda: ChainMap({}, {}, self.defaults))

        self.state: defaultdict[torch.Tensor, dict[str, Any]] = defaultdict(dict)
        self.global_state: dict[str, Any] = {}

        self.children: dict[str, Module] = {}

        self._overridden_keys = set()


    def set_param_groups(self, param_groups: Params):
        param_groups = _make_param_groups(param_groups, differentiable=False)
        for group in param_groups:
            settings = group.copy()
            params = settings.pop('params')
            if not settings: continue
            self._overridden_keys.update(*settings.keys())

            for param in params:
                self.settings[param].maps[0].update(settings) # set module-specific per-parameter settings
        return self

    def set_child(self, key: str, module: "Module | Iterable[Module]"):
        self.children[key] = maybe_chain(module)

    def set_children_sequence(self, modules: "Iterable[Module | Iterable[Module]]", prefix = 'module_'):
        modules = list(modules)
        for i, m in enumerate(modules):
            self.set_child(f'{prefix}{i}', maybe_chain(m))

    def get_children_sequence(self, prefix = 'module_'):
        return [self.children[f'{prefix}{i}'] for i in range(len(self.children)) if f'{prefix}{i}' in self.children]


    @overload
    def get_settings(self, key: str, *,
                     params: Sequence[torch.Tensor] | Vars, cls: type[ListLike] = list) -> ListLike: ...
    @overload
    def get_settings(self, key: list[str] | tuple[str,...], *,
                     params: Sequence[torch.Tensor] | Vars, cls: type[ListLike] = list) -> list[ListLike]: ...
    @overload
    def get_settings(self, key: str, key2: str, *keys: str,
                     params: Sequence[torch.Tensor] | Vars, cls: type[ListLike] = list) -> list[ListLike]: ...

    def get_settings(self, key: str | list[str] | tuple[str,...], key2: str | None = None, *keys: str,
                     params: Sequence[torch.Tensor] | Vars, cls: type[ListLike] = list) -> ListLike | list[ListLike]:
        if isinstance(params, Vars): params = params.params
        return get_state_vals(self.settings, params, key, key2, *keys, must_exist=True, cls=cls) # pyright:ignore[reportArgumentType]


    @overload
    def get_state(self, key: str, *,
                   params: Sequence[torch.Tensor] | Vars, must_exist: bool = False, init: Init = torch.zeros_like,
                   cls: type[ListLike] = list) -> ListLike: ...
    @overload
    def get_state(self, key: list[str] | tuple[str,...], *,
                   params: Sequence[torch.Tensor] | Vars, must_exist: bool = False, init: Init | Sequence[Init] = torch.zeros_like,
                   cls: type[ListLike] = list) -> list[ListLike]: ...
    @overload
    def get_state(self, key: str, key2: str, *keys: str,
                   params: Sequence[torch.Tensor] | Vars, must_exist: bool = False, init: Init | Sequence[Init] = torch.zeros_like,
                   cls: type[ListLike] = list) -> list[ListLike]: ...

    def get_state(self, key: str | list[str] | tuple[str,...], key2: str | None = None, *keys: str,
                   params: Sequence[torch.Tensor] | Vars, must_exist: bool = False, init: Init | Sequence[Init] = torch.zeros_like,
                   cls: type[ListLike] = list) -> ListLike | list[ListLike]:
        """Returns values of per-parameter state for a given key.
        If key doesn't exist, create it with inits.

        This functions like `operator.itemgetter`, returning a single value if called with a single key,
        or tuple of called with multiple keys.

        If you want to force it to return a tuple even with a single key, pass a list/tuple of 1 or more keys.

        .. code:: py

            exp_avg = self.state_vals("exp_avg")
            # returns cls (by default TensorList)

            exp_avg, exp_avg_sq = self.state_vals("exp_avg", "exp_avg_sq")
            # returns list of cls

            exp_avg = self.state_vals(["exp_avg"])
            # always returns a list of cls, even if got a single key


        Args:
            *keys (str):
                the keys to look for in each parameters state.
                if a single key is specified, this returns a single value or cls,
                otherwise this returns a list of values or cls per each key.
            params (Iterable[torch.Tensor]): parameters to return the states for.
            must_exist (bool, optional):
                If a key doesn't exist in state, if True, raises a KeyError, if False, creates the value
                using `init` argument (default = False).
            init (Init | Sequence[Init], optional):
                how to initialize a key if it doesn't exist.

                can be
                - Callable like torch.zeros_like
                - string - "param" or "grad" to use cloned params or cloned grads.
                - anything else other than list/tuples will be used as-is, tensors will be cloned.
                - list/tuple of values per each parameter, only if got a single key.
                - list/tuple of values per each key, only if got multiple keys.

                if multiple `keys` are specified, inits is per-key!

                Defaults to torch.zeros_like.
            cls (type[ListLike], optional):
                MutableSequence class to return, this only has effect when state_keys is a list/tuple. Defaults to list.

        Returns:
            - if state_keys has a single key and keys has a single key, return a single value.
            - if state_keys has a single key and keys has multiple keys, return a list of values.
            - if state_keys has multiple keys and keys has a single key, return cls.
            - if state_keys has multiple keys and keys has multiple keys, return list of cls.
        """
        if isinstance(params, Vars): params = params.params
        return get_state_vals(self.state, params, key, key2, *keys, must_exist=must_exist, init=init, cls=cls) # pyright:ignore[reportArgumentType]

    def state_dict(self):
        """state dict"""
        # TODO
    # ---------------------------- OVERRIDABLE METHODS --------------------------- #
    @abstractmethod
    def step(self, vars: Vars) -> Vars:
        """perform a step, returns new vars but may update them in-place."""

    def reset_stats(self):
        """Resets running stats of this module, by default completely clears per-parameter states and global_state."""
        self.state.clear()
        self.global_state.clear()
# endregion

Chainable = Module | Iterable[Module]


def unroll_modules(*modules: Chainable) -> list[Module]:
    unrolled = []

    for m in modules:
        if isinstance(m, Module):
            unrolled.append(m)
            unrolled.extend(unroll_modules(m.children.values()))
        else:
            unrolled.extend(unroll_modules(*m))

    return unrolled


# region Modular
# ---------------------------------- Modular --------------------------------- #
class Modular(torch.optim.Optimizer):
    # this is specifically for lr schedulers
    param_groups: list[ChainMap[str, Any]] # pyright:ignore[reportIncompatibleVariableOverride]

    def __init__(self, params: Params | torch.nn.Module, *modules: Module):
        self.model = None
        if isinstance(params, torch.nn.Module):
            self.model = params
            params = params.parameters()

        self.modules = modules
        self.unrolled_modules = unroll_modules(self.modules)
        param_groups = _make_param_groups(params, differentiable=False)
        self._per_parameter_global_settings: dict[torch.Tensor, list[MutableMapping[str, Any]]] = {}

        # make sure there is no more than a single learning rate module
        lr_modules = [m for m in self.unrolled_modules if 'lr' in m.defaults]
        if len(lr_modules) > 1:
            warnings.warn(f'multiple learning rate modules detected: {lr_modules}. This may lead to componding of learning rate multiplication with per-parameter learning rates and schedulers.')

        # iterate over all per-parameter settings overrides and check if they are applied at most once
        for group in param_groups:
            for k in group:
                if k in ('params', 'lr'): continue
                modules_with_k = [m for m in self.unrolled_modules if k in m.defaults and k not in m._overridden_keys]
                if len(modules_with_k) > 1:
                    warnings.warn(f'`params` has a `{k}` key, and multiple modules have that key: {modules_with_k}. If you intended to only set `{k}` to one of them, use `module.set_param_groups(params)`')

        # defaults for schedulers
        defaults = {}
        for m in self.unrolled_modules: defaults.update(m.defaults)
        super().__init__(param_groups, defaults=defaults)

        self.current_step = 0

    def add_param_group(self, param_group: dict[str, Any]):
        proc_param_group = _make_param_groups([param_group], differentiable=False)[0]
        self.param_groups.append(ChainMap(proc_param_group, self.defaults))

        for p in proc_param_group['params']:
            # updates global per-parameter setting overrides (medium priority)
            self._per_parameter_global_settings[p] = [m.settings[p].maps[1] for m in self.unrolled_modules]


    def step(self, closure=None):
        # propagate global per-parameter setting overrides
        for g in self.param_groups:
            settings = dict(g.maps[0]) # ignore defaults
            params = settings.pop('params')
            if not settings: continue

            for p in params:
                if not p.requires_grad: continue
                for map in self._per_parameter_global_settings[p]: map.update(settings)

        # create vars
        params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]
        vars = Vars(params=params, closure=closure, model=self.model, current_step=self.current_step)

        # if closure is None, assume backward has been called and gather grads
        if closure is None:
            vars.grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

        last_module = self.modules[-1]
        last_lr = last_module.defaults.get('lr', None)
        n_modules = len(self.modules)

        # step
        for i, module in enumerate(self.modules):

            # last module, or next to last module before lr
            if (i == n_modules - 1) or ((i == n_modules - 2) and (last_lr is not None)):
                if module.children: vars.nested_is_last = True
                else: vars.is_last = True
                if last_lr is not None: vars.last_module_lrs = last_module.get_settings('lr', params=vars)

            vars = module.step(vars)
            if vars.stop: break

        # apply update
        if vars.update is not None:
            with torch.no_grad():
                torch._foreach_sub_(params, vars.update)

        for hook in vars.post_step_hooks:
            hook(self, vars)

        self.current_step += 1
        return vars.loss if vars.loss is not None else vars.loss_approx
# endregion

# region Chain
# ----------------------------------- Chain ---------------------------------- #
class Chain(Module):
    def __init__(self, *modules: Module | Iterable[Module]):
        super().__init__()
        flat_modules: list[Module] = flatten(modules)
        for i, module in enumerate(flat_modules):
            self.set_child(f'module_{i}', module)

    def step(self, vars):
        for i in range(len(self.children)):
            vars = self.children[f'module_{i}'].step(vars)
            if vars.stop: break
        return vars

def maybe_chain(*modules: Chainable) -> Module:
    """if more than 1 modules returns chain else returns module"""
    flat_modules: list[Module] = flatten(modules)
    if len(flat_modules) == 1:
        return flat_modules[0]
    else:
        return Chain(*flat_modules)
# endregion
