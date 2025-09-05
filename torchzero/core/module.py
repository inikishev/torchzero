import warnings
from abc import ABC, abstractmethod
from collections import ChainMap, defaultdict
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from operator import itemgetter
from typing import Any, Literal, cast, final, overload

import torch

from ..utils import (
    Init,
    ListLike,
    Params,
    _make_param_groups,
    get_state_vals,
    vec_to_tensors,
)
from ..utils.derivatives import flatten_jacobian, hvp, hvp_fd_central, hvp_fd_forward
from ..utils.linalg.linear_operator import LinearOperator
from ..utils.python_tools import flatten
from .var import Var


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
        from .chain import maybe_chain
        self.children[key] = maybe_chain(module)

    def set_children_sequence(self, modules: "Iterable[Module | Sequence[Module]]", prefix = 'module_'):
        from .chain import maybe_chain

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

        ```python
        exp_avg = self.state_vals("exp_avg")
        # returns cls (by default TensorList)

        exp_avg, exp_avg_sq = self.state_vals("exp_avg", "exp_avg_sq")
        # returns list of cls

        exp_avg = self.state_vals(["exp_avg"])
        # always returns a list of cls, even if got a single key
        ```

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

    def _load_state_dict(self, state_dict: dict[str, Any], id_to_tensor: dict[int, torch.Tensor]):
        """loads state_dict, ``id_to_tensor`` is passed by ``Modular``"""
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
            if k in self.children: self.children[k]._load_state_dict(v, id_to_tensor)
            else: warnings.warn(f'State dict for {self} has child {k}, which is missing in {self}')

        # extra info
        self._extra_unpack(state_dict['extra'])

    # ---------------------------- OVERRIDABLE METHODS --------------------------- #
    def step(self, var: Var) -> Var:
        """performs a step, returns new ``var`` but may update it in-place."""
        self.update(var)
        return self.apply(var)

    def update(self, var:Var) -> Any:
        """Updates the internal state of this module. This should not modify ``var.update``.

        Specifying ``update`` and ``apply`` methods is optional and allows certain meta-modules to be used,
        such as ``tz.m.Online`` or trust regions. Alternatively, simply override the ``step`` method.
        """

    def apply(self, var: Var) -> Var:
        """Applies this module to ``var.get_update()``.
        This should not modify the internal state of this module if possible.

        Specifying ``update`` and ``apply`` methods is optional and allows certain meta-modules to be used,
        such as ``tz.m.Online`` or trust regions. Alternatively, simply override the ``step`` method.
        """
        return self.step(var)

    def get_H(self, var: Var) -> LinearOperator | None:
        """returns a ``LinearOperator`` corresponding to hessian or hessian approximation.
        The hessian approximation is assumed to be for all parameters concatenated to a vector."""
        # if this method is not defined it searches in children
        # this should be overwritten to return None if child params are different from this modules params
        H = None
        for k,v in self.children.items():
            H_v = v.get_H(var)

            if (H is not None) and (H_v is not None):
                raise RuntimeError(f"Two children of {self} have a hessian, second one is {k}={v}")

            if H_v is not None: H = H_v

        return H

    def reset(self):
        """Resets the internal state of the module (e.g. momentum) and all children. By default clears state and global state."""
        self.state.clear()

        generator = self.global_state.get("generator", None)
        self.global_state.clear()
        if generator is not None: self.global_state["generator"] = generator

        for c in self.children.values(): c.reset()

    def reset_for_online(self):
        """Resets buffers that depend on previous evaluation, such as previous gradient and loss,
        which may become inaccurate due to mini-batching.

        ``Online`` module calls ``reset_for_online``,
        then it calls ``update`` with previous parameters,
        then it calls ``update`` with current parameters,
        and then ``apply``.
        """
        for c in self.children.values(): c.reset_for_online()

    def _extra_pack(self):
        """extra information to store in state_dict of this optimizer.
        Will be passed to ``_extra_unpack`` when loading the state_dict."""
        return {}

    def _extra_unpack(self, x):
        """``_extra_pack`` return will be passed to this method when loading state_dict.
        This method is called after loading the rest of the state dict"""

    def get_generator(self, device: torch.types.Device, seed: int | None):
        if seed is None: return None

        if 'generator' not in self.global_state:
            self.global_state['generator'] = torch.Generator(device).manual_seed(seed)

        return self.global_state['generator']

Chainable = Module | Sequence[Module]
