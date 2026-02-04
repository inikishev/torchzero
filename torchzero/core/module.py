import warnings
from abc import ABC, abstractmethod
from collections import ChainMap, defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, overload, TYPE_CHECKING, Literal

import torch

from ..linalg.linear_operator import LinearOperator
from ..utils.optimizer import Init, ListLike, get_state_vals
from ..utils.params import Params, _make_param_groups, _set_fake_params_, _empty_fake_param_storage_
from .functional import step_tensors

if TYPE_CHECKING:
    from .objective import Objective

ProjectedBuffer = Literal["grad", "grad_sq", "grad_cu", "covariance", "inverse"]

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
        if any(isinstance(v, Module) for v in defaults.values()): raise RuntimeError("Passed a module to defaults")
        self.defaults: dict[str, Any] = defaults

        # settings are stored like state in per-tensor defaultdict, with per-parameter overrides possible
        # 0 - this module specific per-parameter setting overrides set via `set_param_groups` - highest priority
        # 1 - global per-parameter setting overrides in param_groups passed to Optimizer - medium priority
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
        """tracks keys overridden with ``set_param_groups``, only used to not give a warning"""

        self._projected_keys: defaultdict[ProjectedBuffer, set[str]] = defaultdict(set)
        """tracks keys with gradient-like buffers, covariance-like buffers, etc for reprojecting"""

        self._fake_params: dict[str, list[torch.Tensor]] = {}
        """fake parameters for state keys and shape inference, key is name of child, value is list of fake parameters"""


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

    def set_child(self, key: str, module: "Module | Sequence[Module] | None"):
        if key in self.children:
            warnings.warn(f"set_child overwriting child `{key}`")

        if module is None: return

        from .chain import maybe_chain
        self.children[key] = maybe_chain(module)

    def set_children_sequence(self, modules: "Iterable[Module | Sequence[Module]]", prefix = 'module_'):
        from .chain import maybe_chain

        modules = list(modules)
        for i, m in enumerate(modules):
            self.set_child(f'{prefix}{i}', maybe_chain(m))

    def get_children_sequence(self, prefix = 'module_'):
        return [self.children[f'{prefix}{i}'] for i in range(len(self.children)) if f'{prefix}{i}' in self.children]

    def inner_step(
        self,
        key: str,
        objective: "Objective",
        must_exist: bool = True,
    ) -> "Objective":
        """Passes ``objective`` to child and returns it."""
        child = self.children.get(key, None)

        if child is None:
            if must_exist: raise KeyError(f"child `{key}` doesn't exist")
            return objective

        return child.step(objective)


    def inner_step_tensors(
        self,
        key: str,
        tensors: list[torch.Tensor],
        clone: bool,
        params: Iterable[torch.Tensor] | None = None,
        grads: Sequence[torch.Tensor] | None = None,
        loss: torch.Tensor | None = None,
        closure: Callable | None = None,
        objective: "Objective | None" = None,
        must_exist: bool = True
    ) -> list[torch.Tensor]:
        """Steps with child module. Can be used to apply transforms to any internal buffers.

        If ``objective`` is specified, other attributes shouldn't to be specified.

        Args:
            key (str): Child module key.
            tensors (Sequence[torch.Tensor]): tensors to pass to child module.
            clone (bool):
                If ``key`` exists, whether to clone ``tensors`` to avoid modifying buffers in-place.
                If ``key`` doesn't exist, ``tensors`` are always returned without cloning
            params (Iterable[torch.Tensor] | None, optional):
                pass None if ``tensors`` have different shape, it will create fake params from tensors
                for state keys and shape inference. Defaults to None.
            grads (Sequence[torch.Tensor] | None, optional): grads. Defaults to None.
            loss (torch.Tensor | None, optional): loss. Defaults to None.
            closure (Callable | None, optional): closure. Defaults to None.
            must_exist (bool, optional): if True, if ``key`` doesn't exist, raises ``KeyError``. Defaults to True.
        """

        child = self.children.get(key, None)

        if child is None:
            if must_exist: raise KeyError(f"child `{key}` doesn't exist")
            return tensors

        if clone: tensors = [t.clone() for t in tensors]

        # set fake params to same storage as tensors so as to not use any extra memory
        # while they still refer to same python objects, so they can be used
        # as state keys and for shape inference when params aren't given.
        fake = params is None
        if fake:
            if key not in self._fake_params:
                self._fake_params[key] = [torch.empty_like(t) for t in tensors]
            params = self._fake_params[key]
            _set_fake_params_(params, tensors)

        update = step_tensors(modules=child, tensors=tensors, params=params, grads=grads,
                            loss=loss, closure=closure, objective=objective)

        # set fake params storage to empty
        if fake:
            _empty_fake_param_storage_(params)

        return update


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
        return get_state_vals(self.state, params, key, key2, *keys, must_exist=must_exist, init=init, cls=cls) # pyright:ignore[reportArgumentType]

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
        """loads state_dict, ``id_to_tensor`` is passed by ``Optimizer``"""
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

    def get_generator(self, device: torch.types.Device, seed: int | None):
        """If ``seed=None``, returns ``None``.

        Otherwise, if generator on this device and with this seed hasn't been created,
        creates it and stores in global state.

        Returns ``torch.Generator``."""
        if seed is None: return None

        if device is None: device_obj = torch.get_default_device()
        else: device_obj = torch.device(device)
        key = f"__generator-{seed}-{device_obj.type}:{device_obj.index}"

        if key not in self.global_state:
            self.global_state[key] = torch.Generator(device).manual_seed(seed)

        return self.global_state[key]

    def increment_counter(self, key: str, start: int):
        """first value is ``start``"""
        value = self.global_state.get(key, start - 1) + 1
        self.global_state[key] = value
        return value

    def get_child_projected_buffers(self, key: str, buff: ProjectedBuffer | Sequence[ProjectedBuffer], params:Sequence[torch.Tensor] | None = None) -> list[list[torch.Tensor]]:
        """if params is None, assumes fake parameters"""
        if isinstance(buff, str): buff = (buff, )

        child = self.children[key]
        child.on_get_projected_buffers()
        if params is None:
            params = self._fake_params[key]

        vals = []
        for b in buff:
            for buff_key in child._projected_keys[b]:
                state = child.state[params[0]]
                if buff_key in state:
                    tensors = [child.state[p][buff_key] for p in params]
                    if isinstance(tensors[0], torch.Tensor):
                        vals.append(tensors)
                    else: # its usually a deque
                        assert isinstance(tensors[0], Sequence), type(tensors[0])
                        vals.extend(zip(*tensors))

                elif buff_key in child.global_state:
                    val = child.global_state[buff_key]
                    if len(val) == 0: continue
                    if isinstance(val[0], torch.Tensor):
                        vals.append(val)
                    else:
                        assert isinstance(val[0], Sequence)
                        vals.extend(zip(*vals))

        # recursively do this on children,
        # note that if params are fake, children will have same fake params
        # unless that child steps with something else. I don't think that is feasible to support it
        for c in child.children:
            vals.extend(child.get_child_projected_buffers(c, buff, params=params))

        return vals

    def add_projected_keys(self, buffer: ProjectedBuffer, *keys):
        for k in keys: self._projected_keys[buffer].add(k)


    # ---------------------------- OVERRIDABLE METHODS --------------------------- #
    def update(self, objective:"Objective") -> None:
        """Updates internal state of this module. This should not modify ``objective.update``.

        Specifying ``update`` and ``apply`` methods is optional and allows certain meta-modules to be used,
        such as ``tz.m.Online`` or trust regions. Alternatively, define all logic within the ``apply`` method.

        ``update`` is guaranteed to be called at least once before ``apply``.

        Args:
            objective (Objective): ``Objective`` object
        """

    @abstractmethod
    def apply(self, objective: "Objective") -> "Objective":
        """Updates ``objective`` using the internal state of this module.

        If ``update`` method is defined, ``apply`` shouldn't modify the internal state of this module if possible.

        Specifying ``update`` and ``apply`` methods is optional and allows certain meta-modules to be used,
        such as ``tz.m.Online`` or trust regions. Alternatively, define all logic within the ``apply`` method.

        ``update`` is guaranteed to be called at least once before ``apply``.

        Args:
            objective (Objective): ``Objective`` object
        """
        # if apply is empty, it should be defined explicitly.
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement `apply`.")

    def step(self, objective: "Objective") -> "Objective":
        """Perform a step with this module. Calls ``update``, then ``apply``."""
        self.update(objective)
        return self.apply(objective)

    def get_H(self, objective: "Objective") -> LinearOperator | None:
        """returns a ``LinearOperator`` corresponding to hessian or hessian approximation.
        The hessian approximation is assumed to be for all parameters concatenated to a vector."""
        # if this method is not defined it searches in children
        # this should be overwritten to return None if child params are different from this modules params
        H = None
        for k,v in self.children.items():
            H_v = v.get_H(objective)

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

    def on_get_projected_buffers(self):
        """runs before projected buffers are accessed"""

    def _extra_pack(self) -> dict:
        """extra information to store in ``state_dict`` of this optimizer.
        Will be passed to ``_extra_unpack`` when loading the ``state_dict``."""
        return {}

    def _extra_unpack(self, d: dict):
        """``_extra_pack`` return will be passed to this method when loading ``state_dict``.
        This method is called after loading the rest of the state dict"""


Chainable = Module | Sequence[Module]
