import warnings
from abc import ABC, abstractmethod
from collections import ChainMap, defaultdict
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from operator import itemgetter
from typing import Any, final, overload, Literal

import torch

from ..utils import (
    Init,
    ListLike,
    Params,
    _make_param_groups,
    get_state_vals,
)
from ..utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward
from ..utils.python_tools import flatten


def _closure_backward(closure, params, retain_graph, create_graph):
    with torch.enable_grad():
        if not (retain_graph or create_graph):
            return closure()

        for p in params: p.grad = None
        loss = closure(False)
        grad = torch.autograd.grad(loss, params, retain_graph=retain_graph, create_graph=create_graph)
        for p,g in zip(params,grad): p.grad = g
        return loss

# region Vars
# ----------------------------------- var ----------------------------------- #
class Var:
    """
    Holds the state and context passed between optimizer modules during a step.

    This class acts as a mutable container for information relevant to the current
    optimization step, such as parameters, gradients, loss, and the computed update.
    Modules read from and write to this object to coordinate their actions.
    """
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
        """gradient with current parameters. If closure is not None, this is set to None and can be calculated if needed."""

        self.loss: torch.Tensor | Any | None = None
        """loss with current parameters."""

        self.loss_approx: torch.Tensor | Any | None = None
        """loss at a point near current point. This can be useful as some modules only calculate loss at perturbed points,
        whereas some other modules require loss strictly at current point."""

        self.post_step_hooks: list[Callable[[Modular, Var]]] = []
        """list of functions to be called after optimizer step.
        The signature is:

        .. code:: py

            def hook(optimizer: Modular, var: Vars): ...

        """

        self.is_last: bool = False
        """
        Indicates that current module is either last or next-to-last before a learning rate module.
        This is always False if current module has children or is a child.
        """

        self.nested_is_last: bool = False
        """
        Indicates that current module is either last or next-to-last before a learning rate module, for modules
        that have children.
        """

        self.last_module_lrs: list[float] | None = None
        """
        List of per-parameter learning rates if current module is next-to-last before a
        learning rate module, otherwise this is set to None. Ignore this unless you are manually applying
        update to parameters.
        """

        self.stop: bool = False
        """if True, all following modules will be skipped."""

        self.skip_update: bool = False
        """if True, the parameters will not be updated"""

        self.storage: dict = {}
        """Storage for any other data, such as hessian estimates, etc"""

    def get_loss(self, backward: bool, retain_graph = None, create_graph: bool = False) -> torch.Tensor | float:
        """Returns the loss at current parameters, computing it if it hasn't been computed already and assigning :code:`var.loss`.
        Do not call this at perturbed parameters. Backward always zeroes grads before recomputing."""

        if self.loss is None:
            if self.closure is None: raise RuntimeError("closure is None")
            if backward:
                with torch.enable_grad():
                    self.loss = self.loss_approx = _closure_backward(
                        closure=self.closure, params=self.params, retain_graph=retain_graph, create_graph=create_graph
                    )

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
                self.loss = self.loss_approx = _closure_backward(
                    closure=self.closure, params=self.params, retain_graph=retain_graph, create_graph=create_graph
                )
            self.grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        return self.loss # type:ignore

    def get_grad(self, retain_graph: bool | None = None, create_graph: bool = False) -> list[torch.Tensor]:
        """Returns the gradient at initial parameters, computing it if it hasn't been computed already and assigning
        :code:`var.grad` and potentially :code:`var.loss`. Do not call this at perturbed parameters."""
        if self.grad is None:
            if self.closure is None: raise RuntimeError("closure is None")
            self.get_loss(backward=True, retain_graph=retain_graph, create_graph=create_graph) # evaluate and set self.loss and self.grad

        assert self.grad is not None
        return self.grad

    def get_update(self) -> list[torch.Tensor]:
        """Returns the update. If update is None, it is initialized by cloning the gradients and assigning to :code:`var.update`.
        Computing the gradients may assign :code:`var.grad` and :code:`var.loss` if they haven't been computed.
        Do not call this at perturbed parameters."""
        if self.update is None: self.update = [g.clone() for g in self.get_grad()]
        return self.update

    def clone(self, clone_update: bool):
        """Creates a shallow copy of the Vars object, update can optionally be deep-copied (via :code:`torch.clone`)."""
        copy = Var(params = self.params, closure=self.closure, model=self.model, current_step=self.current_step)

        if clone_update and self.update is not None:
            copy.update = [u.clone() for u in self.update]
        else:
            copy.update = self.update

        copy.grad = self.grad
        copy.loss = self.loss
        copy.loss_approx = self.loss_approx
        copy.post_step_hooks = self.post_step_hooks
        copy.stop = self.stop
        copy.skip_update = self.skip_update

        return copy

    def update_attrs_from_clone_(self, var: "Var"):
        """Updates attributes of this `Vars` instance from a cloned instance.
        Typically called after a child module has processed a cloned `Vars`
        object. This propagates any newly computed loss or gradient values
        from the child's context back to the parent `Vars` if the parent
        didn't have them computed already.
        """
        if self.loss is None: self.loss = var.loss
        if self.loss_approx is None: self.loss_approx = var.loss_approx
        if self.grad is None: self.grad = var.grad
        self.storage.update(var.storage)

    def zero_grad(self, set_to_none=True):
        if set_to_none:
            for p in self.params: p.grad = None
        else:
            grads = [p.grad for p in self.params if p.grad is not None]
            if len(grads) != 0: torch._foreach_zero_(grads)

# endregion

# region Module
# ---------------------------------- module ---------------------------------- #
class Module(ABC):
    """Abstract base class for an optimizer modules.

    Modules represent distinct steps or transformations within the optimization
    process (e.g., momentum, line search, gradient accumulation).

    A module does not store parameters, but it maintains per-parameter state and per-parameter settings
    where tensors are used as keys (same as torch.optim.Optimizer state.)

    Args:
        defaults (dict[str, Any] | None):
            a dict containing default values of optimization options (used when a parameter group doesn't specify them).
"""
    def __init__(self, defaults: dict[str, Any] | None = None):
        if defaults is None: defaults = {}
        self.defaults: dict[str, Any] = defaults

        # settings are stored like state in per-tensor defaultdict, with per-parameter overrides possible
        # 0 - this module specific per-parameter setting overrides set via `set_param_groups` - highest priority
        # 1 - global per-parameter setting overrides in param_groups passed to Modular - medium priority
        # 2 - `defaults` - lowest priority
        self.settings: defaultdict[torch.Tensor, ChainMap[str, Any]] = defaultdict(lambda: ChainMap({}, {}, self.defaults))
        """per-parameter settings."""

        self.state: defaultdict[torch.Tensor, dict[str, Any]] = defaultdict(dict)
        """Per-parameter state (e.g., momentum buffers)."""

        self.global_state: dict[str, Any] = {}
        """Global state for things that are not per-parameter."""

        self.children: dict[str, Module] = {}
        """A dictionary of child modules."""

        self._overridden_keys = set()
        """tracks keys overridden with `set_param_groups`, only used to not give a warning"""


    def set_param_groups(self, param_groups: Params):
        """Set custom parameter groups with per-parameter settings that this module will use."""
        param_groups = _make_param_groups(param_groups, differentiable=False)
        for group in param_groups:
            settings = group.copy()
            params = settings.pop('params')
            if not settings: continue
            self._overridden_keys.update(*settings.keys())

            for param in params:
                self.settings[param].maps[0].update(settings) # set module-specific per-parameter settings
        return self

    def set_child(self, key: str, module: "Module | Sequence[Module]"):
        self.children[key] = maybe_chain(module)

    def set_children_sequence(self, modules: "Iterable[Module | Sequence[Module]]", prefix = 'module_'):
        modules = list(modules)
        for i, m in enumerate(modules):
            self.set_child(f'{prefix}{i}', maybe_chain(m))

    def get_children_sequence(self, prefix = 'module_'):
        return [self.children[f'{prefix}{i}'] for i in range(len(self.children)) if f'{prefix}{i}' in self.children]

    def __repr__(self):
        s = self.__class__.__name__
        if self.children:
            s = f'{s}('
            for k,v in self.children.items():
                s = f'{s}{k}={v}, '
            s = f'{s[:-2]})'
        return s

    @overload
    def get_settings(self, params: Sequence[torch.Tensor], key: str, *,
                     cls: type[ListLike] = list) -> ListLike: ...
    @overload
    def get_settings(self, params: Sequence[torch.Tensor], key: list[str] | tuple[str,...], *,
                     cls: type[ListLike] = list) -> list[ListLike]: ...
    @overload
    def get_settings(self, params: Sequence[torch.Tensor], key: str, key2: str, *keys: str,
                     cls: type[ListLike] = list) -> list[ListLike]: ...

    def get_settings(self, params: Sequence[torch.Tensor], key: str | list[str] | tuple[str,...], key2: str | None = None,
                     *keys: str, cls: type[ListLike] = list) -> ListLike | list[ListLike]:
        # if isinstance(params, Vars): params = params.params
        return get_state_vals(self.settings, params, key, key2, *keys, must_exist=True, cls=cls) # pyright:ignore[reportArgumentType]


    @overload
    def get_state(self, params: Sequence[torch.Tensor], key: str, *,
                   must_exist: bool = False, init: Init = torch.zeros_like,
                   cls: type[ListLike] = list) -> ListLike: ...
    @overload
    def get_state(self, params: Sequence[torch.Tensor], key: list[str] | tuple[str,...], *,
                   must_exist: bool = False, init: Init | Sequence[Init] = torch.zeros_like,
                   cls: type[ListLike] = list) -> list[ListLike]: ...
    @overload
    def get_state(self, params: Sequence[torch.Tensor], key: str, key2: str, *keys: str,
                   must_exist: bool = False, init: Init | Sequence[Init] = torch.zeros_like,
                   cls: type[ListLike] = list) -> list[ListLike]: ...

    def get_state(self, params: Sequence[torch.Tensor], key: str | list[str] | tuple[str,...], key2: str | None = None, *keys: str,
                   must_exist: bool = False, init: Init | Sequence[Init] = torch.zeros_like,
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
        # if isinstance(params, Vars): params = params.params
        return get_state_vals(self.state, params, key, key2, *keys, must_exist=must_exist, init=init, cls=cls) # pyright:ignore[reportArgumentType]

    # def first_setting(self, *keys:str, params:Sequence[torch.Tensor]):
    #     # if isinstance(params, Vars): params = params.params
    #     return itemgetter(*keys)(self.settings[params[0]])

    def clear_state_keys(self, *keys:str):
        for s in self.state.values():
            for k in keys:
                if k in s: del s[k]

    @overload
    def store(self, params: Sequence[torch.Tensor], keys: str, values: Sequence): ...
    @overload
    def store(self, params: Sequence[torch.Tensor], keys: Sequence[str], values: Sequence[Sequence]): ...
    def store(self, params: Sequence[torch.Tensor], keys: str | Sequence[str], values: Sequence):
        if isinstance(keys, str):
            for p,v in zip(params, values):
                state = self.state[p]
                state[keys] = v
            return

        for p, *p_v in zip(params, *values):
            state = self.state[p]
            for k,v in zip(keys, p_v): state[k] = v

    def state_dict(self):
        """state dict"""
        packed_state = {id(k):v for k,v in self.state.items()}
        packed_settings = {id(k):v for k,v in self.settings.items()}

        state_dict = {
            "state": packed_state,
            "settings":
                {
                    "local": {k:v.maps[0] for k,v in packed_settings.items()},
                    "global": {k:v.maps[1] for k,v in packed_settings.items()},
                    "defaults": {k:v.maps[2] for k,v in packed_settings.items()},
                },
            "global_state": self.global_state,
            "extra": self._extra_pack(),
            "children": {k: v.state_dict() for k, v in self.children.items()}
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any], id_to_tensor: dict[int, torch.Tensor]):
        # load state
        state = state_dict['state']
        self.state.clear()
        self.state.update({id_to_tensor[k]:v for k,v in state.items()})

        # load settings
        settings = state_dict['settings']
        self.settings.clear()
        for k, v in settings['local'].items(): self.settings[id_to_tensor[k]].maps[0].update(v)
        for k, v in settings['global'].items(): self.settings[id_to_tensor[k]].maps[1].update(v)
        for k, v in settings['defaults'].items(): self.settings[id_to_tensor[k]].maps[2].update(v)

        # load global state
        self.global_state.clear()
        self.global_state.update(state_dict['global_state'])

        # children
        for k, v in state_dict['children']:
            if k in self.children: self.children[k].load_state_dict(v, id_to_tensor)
            else: warnings.warn(f'State dict for {self} has child {k}, which is missing in {self}')

        # extra info
        self._extra_unpack(state_dict['extra'])

    # ---------------------------- OVERRIDABLE METHODS --------------------------- #
    def step(self, var: Var) -> Var:
        """performs a step, returns new var but may update it in-place."""
        self.update(var)
        return self.apply(var)

    def update(self, var:Var) -> Any:
        """Updates the internal state of this module. This should not modify `var.update`.

        Specifying ``update`` and ``apply`` methods is optional and allows certain meta-modules to be used,
        such as ::code::`tz.m.Online`.
        """

    def apply(self, var: Var) -> Var:
        """Applies this module to ``var.get_update()``. This should not modify the internal state of this module if possible."""
        raise NotImplementedError(f"{self} doesn't implement the `apply` method.")

    def reset(self):
        """Resets the internal state of the module (e.g. momentum). By default clears state and global state."""
        # no complex logic is allowed there because this is overridden by many modules
        # where super().reset() shouldn't be called
        self.state.clear()
        self.global_state.clear()

    def reset_for_online(self):
        """resets only the intermediate state of this module, e.g. previous parameters and gradient."""
        for c in self.children.values(): c.reset_for_online()

    def _extra_pack(self):
        return {}

    def _extra_unpack(self, x):
        pass


    # ------------------------------ HELPER METHODS ------------------------------ #
    @torch.no_grad
    def Hvp(
        self,
        v: Sequence[torch.Tensor],
        at_x0: bool,
        var: Var,
        rgrad: Sequence[torch.Tensor] | None,
        hvp_method: Literal['autograd', 'forward', 'central'],
        h: float,
        normalize: bool,
        retain_grad: bool,
    ):
        """
        Returns ``(Hvp, rgrad)``. ``rgrad`` is gradient at current parameters, possibly with create_graph=True, or it may be None with ``hvp_method="central"``. Gradient is set to vars automatically if ``at_x0``, you can always access it with ``vars.get_grad()``

        Single sample example:

        .. code:: py

            Hvp, _ = self.hvp(v, at_x0=True, rgrad=None, ..., retain_graph=False)

        Multiple samples example:

        .. code:: py

            D = None
            rgrad = None
            for i in range(n_samples):
                v = [torch.randn_like(p) for p in params]
                Hvp, rgrad = self.hvp(v, at_x0=True, rgrad=rgrad, ..., retain_graph=i < n_samples-1)

                if D is None: D = Hvp
                else: torch._foreach_add_(D, Hvp)

            if n_samples > 1: torch._foreach_div_(D, n_samples)
        Args:
            v (Sequence[torch.Tensor]): vector in hessian-vector product
            at_x0 (bool): whether this is being called at original or perturbed parameters.
            var (Var): Var
            rgrad (Sequence[torch.Tensor] | None): pass None initially, then pass what this returns.
            hvp_method (str): hvp method.
            h (float): finite difference step size
            normalize (bool): whether to normalize v for finite difference
            retain_grad (bool): retain grad
        """
        # get grad
        if rgrad is None and hvp_method in ('autograd', 'forward'):
            if at_x0: rgrad = var.get_grad(create_graph = hvp_method=='autograd')
            else:
                if var.closure is None: raise RuntimeError("Closure is required to calculate HVp")
                with torch.enable_grad():
                    loss = var.closure()
                    rgrad = torch.autograd.grad(loss, var.params, create_graph = hvp_method=='autograd')

        if hvp_method == 'autograd':
            assert rgrad is not None
            Hvp = hvp(var.params, rgrad, v, retain_graph=retain_grad)

        elif hvp_method == 'forward':
            assert rgrad is not None
            loss, Hvp = hvp_fd_forward(var.closure, var.params, v, h=h, g_0=rgrad, normalize=normalize)

        elif hvp_method == 'central':
            loss, Hvp = hvp_fd_central(var.closure, var.params, v, h=h, normalize=normalize)

        else:
            raise ValueError(hvp_method)

        return Hvp, rgrad

# endregion

Chainable = Module | Sequence[Module]


def unroll_modules(*modules: Chainable) -> list[Module]:
    unrolled = []

    for m in modules:
        if isinstance(m, Module):
            unrolled.append(m)
            unrolled.extend(unroll_modules(list(m.children.values())))
        else:
            unrolled.extend(unroll_modules(*m))

    return unrolled


# region Modular
# ---------------------------------- Modular --------------------------------- #

class _EvalCounterClosure:
    """keeps track of how many times closure has been evaluated"""
    __slots__ = ("modular", "closure")
    def __init__(self, modular: "Modular", closure):
        self.modular = modular
        self.closure = closure

    def __call__(self, *args, **kwargs):
        if self.closure is None:
            raise RuntimeError("One of the modules requires closure to be passed to the step method")

        self.modular.num_evaluations += 1
        return self.closure(*args, **kwargs)

# have to inherit from Modular to support lr schedulers
# although Accelerate doesn't work due to converting param_groups to a dict
class Modular(torch.optim.Optimizer):
    """Chains multiple modules into an optimizer.

    Args:
        params (Params | torch.nn.Module): An iterable of parameters to optimize
            (typically `model.parameters()`), an iterable of parameter group dicts,
            or a `torch.nn.Module` instance.
        *modules (Module): A sequence of `Module` instances that define the
            optimization algorithm steps.
    """
    # this is specifically for lr schedulers
    param_groups: list[ChainMap[str, Any]] # pyright:ignore[reportIncompatibleVariableOverride]

    def __init__(self, params: Params | torch.nn.Module, *modules: Module):
        self.model: torch.nn.Module | None = None
        """The model whose parameters are being optimized, if a model instance was passed to `__init__`."""
        if isinstance(params, torch.nn.Module):
            self.model = params
            params = params.parameters()

        self.modules = modules
        """Top-level modules providedduring initialization."""

        self.unrolled_modules = unroll_modules(self.modules)
        """A flattened list of all modules including all children."""

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

        # note - this is what super init does:

        # self.defaults = defaults
        # for param_group in param_groups:
        #     self.add_param_group(param_group)

        self.current_step = 0
        """global step counter for the optimizer."""

        self.num_evaluations = 0
        """number of times the objective has been evaluated (number of closure calls or number of steps if closure is None)."""

    def add_param_group(self, param_group: dict[str, Any]):
        proc_param_group = _make_param_groups([param_group], differentiable=False)[0]
        self.param_groups.append(ChainMap(proc_param_group, self.defaults))

        for p in proc_param_group['params']:
            # updates global per-parameter setting overrides (medium priority)
            self._per_parameter_global_settings[p] = [m.settings[p].maps[1] for m in self.unrolled_modules]

    def state_dict(self):
        all_params = [p for g in self.param_groups for p in g['params']]
        id_to_idx = {id(p): i for i,p in enumerate(all_params)}

        groups = []
        for g in self.param_groups:
            g = g.copy()
            g['params'] = [id_to_idx[id(p)] for p in g['params']]
            groups.append(g)

        state_dict = {
            "idx_to_id": {v:k for k,v in id_to_idx.items()},
            "params": all_params,
            "groups": groups,
            "defaults": self.defaults,
            "modules": {i: m.state_dict() for i, m in enumerate(self.unrolled_modules)}
        }
        return state_dict

    def load_state_dict(self, state_dict: dict):
        self.defaults.clear()
        self.defaults.update(state_dict['defaults'])

        idx_to_param = dict(enumerate(state_dict['params']))
        groups = []
        for g in state_dict['groups']:
            g = g.copy()
            g['params'] = [idx_to_param[p] for p in g['params']]
            groups.append(g)

        self.param_groups.clear()
        for group in groups:
            self.add_param_group(group)

        id_to_tensor = {state_dict['idx_to_id'][i]: p for i,p in enumerate(state_dict['params'])}
        for m, sd in zip(self.unrolled_modules, state_dict['modules'].values()):
            m.load_state_dict(sd, id_to_tensor)


    def step(self, closure=None): # pyright: ignore[reportIncompatibleMethodOverride]
        # propagate global per-parameter setting overrides
        for g in self.param_groups:
            settings = dict(g.maps[0]) # ignore defaults
            params = settings.pop('params')
            if not settings: continue

            for p in params:
                if not p.requires_grad: continue
                for map in self._per_parameter_global_settings[p]: map.update(settings)

        # create var
        params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]
        var = Var(params=params, closure=_EvalCounterClosure(self, closure), model=self.model, current_step=self.current_step)

        # if closure is None, assume backward has been called and gather grads
        if closure is None:
            var.grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            self.num_evaluations += 1

        last_module = self.modules[-1]
        last_lr = last_module.defaults.get('lr', None)
        n_modules = len(self.modules)

        # step
        for i, module in enumerate(self.modules):
            if i!=0: var = var.clone(clone_update=False)

            # last module, or next to last module before lr
            if (i == n_modules - 1) or ((i == n_modules - 2) and (last_lr is not None)):
                if module.children: var.nested_is_last = True
                else: var.is_last = True
                if last_lr is not None: var.last_module_lrs = [last_module.settings[p]['lr'] for p in var.params]

            var = module.step(var)
            if var.stop: break

        # apply update
        if not var.skip_update:
            with torch.no_grad():
                torch._foreach_sub_(params, var.get_update())

        for hook in var.post_step_hooks:
            hook(self, var)

        self.current_step += 1
        return var.loss if var.loss is not None else var.loss_approx

    def __repr__(self):
        return f'Modular({", ".join(str(m) for m in self.modules)})'
# endregion

# region Chain
# ----------------------------------- Chain ---------------------------------- #
class Chain(Module):
    """Chain of modules, mostly used internally"""
    def __init__(self, *modules: Module | Iterable[Module]):
        super().__init__()
        flat_modules: list[Module] = flatten(modules)
        for i, module in enumerate(flat_modules):
            self.set_child(f'module_{i}', module)

    def step(self, var):
        for i in range(len(self.children)):
            var = self.children[f'module_{i}'].step(var)
            if var.stop: break
        return var

    def __repr__(self):
        s = self.__class__.__name__
        if self.children:
            if s == 'Chain': s = 'C' # to shorten it
            s = f'{s}({", ".join(str(m) for m in self.children.values())}'
        return s

def maybe_chain(*modules: Chainable) -> Module:
    """Returns a single module directly if only one is provided, otherwise wraps them in a :code:`Chain`."""
    flat_modules: list[Module] = flatten(modules)
    if len(flat_modules) == 1:
        return flat_modules[0]
    return Chain(*flat_modules)
# endregion

