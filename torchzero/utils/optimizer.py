from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, MutableSequence, Sequence, MutableMapping
from typing import Any, Literal, TypeVar, overload

import torch

from .tensorlist import TensorList
from .numberlist import NumberList
from .torch_tools import tofloat, totensor

ListLike = TypeVar('ListLike', bound=MutableSequence)

ParamFilter = Literal["has_grad", "requires_grad", "all"] | Callable[[torch.Tensor], bool]
def _param_filter(param: torch.Tensor, mode: ParamFilter):
    if callable(mode): return mode(param)
    if mode == 'has_grad': return param.grad is not None
    if mode == 'requires_grad': return param.requires_grad
    if mode == 'all': return True
    raise ValueError(f"Unknown mode {mode}")

def get_params(
    param_groups: Iterable[Mapping[str, Any]],
    mode: ParamFilter = 'requires_grad',
    cls: type[ListLike] = TensorList,
) -> ListLike:
    return cls(p for g in param_groups for p in g['params'] if _param_filter(p, mode)) # type:ignore[reportCallIssue]


@overload
def get_group_vals(param_groups: Iterable[Mapping[str, Any]], key: str, *,
                   mode: ParamFilter = 'requires_grad', cls: type[ListLike] = list) -> ListLike: ...
@overload
def get_group_vals(param_groups: Iterable[Mapping[str, Any]], key: list[str] | tuple[str,...], *,
                   mode: ParamFilter = 'requires_grad', cls: type[ListLike] = list) -> list[ListLike]: ...
@overload
def get_group_vals(param_groups: Iterable[Mapping[str, Any]], key: str, key2: str, *keys: str,
                   mode: ParamFilter = 'requires_grad', cls: type[ListLike] = list) -> list[ListLike]: ...

def get_group_vals(param_groups: Iterable[Mapping[str, Any]],
                   key: str | list[str] | tuple[str,...], key2: str | None = None, *keys: str,
                   mode: ParamFilter = 'requires_grad', cls: type[ListLike] = list) -> ListLike | list[ListLike]:

    # single key, return single cls
    if isinstance(key, str) and key2 is None:
        values = cls()
        for group in param_groups:
            num_params = len([p for p in group['params'] if _param_filter(p, mode)])
            if num_params > 0:
                group_value = group[key]
                values.extend(group_value for _ in range(num_params))
        return values

    # multiple keys
    k1 = (key,) if isinstance(key, str) else tuple(key)
    k2 = () if key2 is None else (key2,)
    keys = k1 + k2 + keys

    values = [cls() for _ in keys]
    for group in param_groups:
        num_params = len([p for p in group['params'] if _param_filter(p, mode)])
        if num_params > 0:
            for i,key in enumerate(keys):
                group_value = group[key]
                values[i].extend(group_value for _ in range(num_params))
    return values

_InitLiterals = Literal['param', 'grad']
Init = _InitLiterals | Any | list[_InitLiterals | Any] | tuple[_InitLiterals | Any]

def _make_initial_state_value(param: torch.Tensor, init: Init, i: int | None):
    if callable(init): return init(param)
    if isinstance(init, torch.Tensor): return init.detach().clone()

    if isinstance(init, str):
        if init in ('param','params'): return param.detach().clone()
        if init in ('grad', 'grads'):
            if param.grad is None: raise RuntimeError('init is set to "grad, but param.grad is None"')
            return param.grad.detach().clone()

    if isinstance(init, (list,tuple)):
        if i is None: raise RuntimeError(f'init is per-parameter ({type(init)}) but parameter index i is None')
        return _make_initial_state_value(param, init[i], None)

    return init

@overload
def get_state_vals(state: Mapping[torch.Tensor, MutableMapping[str, Any]], params: Sequence[torch.Tensor],
                   key: str, *,
                   must_exist: bool = False, init: Init = torch.zeros_like,
                   cls: type[ListLike] = list) -> ListLike: ...
@overload
def get_state_vals(state: Mapping[torch.Tensor, MutableMapping[str, Any]], params: Sequence[torch.Tensor],
                   key: list[str] | tuple[str,...], *,
                   must_exist: bool = False, init: Init | Sequence[Init] = torch.zeros_like,
                   cls: type[ListLike] = list) -> list[ListLike]: ...
@overload
def get_state_vals(state: Mapping[torch.Tensor, MutableMapping[str, Any]], params: Sequence[torch.Tensor],
                   key: str,  key2: str, *keys: str,
                   must_exist: bool = False, init: Init | Sequence[Init] = torch.zeros_like,
                   cls: type[ListLike] = list) -> list[ListLike]: ...

def get_state_vals(state: Mapping[torch.Tensor, MutableMapping[str, Any]], params: Sequence[torch.Tensor],
                   key: str | list[str] | tuple[str,...], key2: str | None = None,  *keys: str,
                   must_exist: bool = False, init: Init | Sequence[Init] = torch.zeros_like,
                   cls: type[ListLike] = list) -> ListLike | list[ListLike]:

    # single key, return single cls
    if isinstance(key, str) and key2 is None:
        values = cls()
        for i, param in enumerate(params):
            s = state[param]
            if key not in s:
                if must_exist: raise KeyError(f"Key {key} doesn't exist in state with keys {tuple(s.keys())}")
                s[key] = _make_initial_state_value(param, init, i)
            values.append(s[key])
        return values

    # multiple keys
    k1 = (key,) if isinstance(key, str) else tuple(key)
    k2 = () if key2 is None else (key2,)
    keys = k1 + k2 + keys

    values = [cls() for _ in keys]
    for i, param in enumerate(params):
        s = state[param]
        for k_i, key in enumerate(keys):
            if key not in s:
                if must_exist: raise KeyError(f"Key {key} doesn't exist in state with keys {tuple(s.keys())}")
                k_init = init[k_i] if isinstance(init, (list,tuple)) else init
                s[key] = _make_initial_state_value(param, k_init, i)
            values[k_i].append(s[key])

    return values


class Optimizer(torch.optim.Optimizer, ABC):
    """subclass of torch.optim.Optimizer with some helper methods for fast experimentation, it's not used anywhere in torchzero.

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults (dict | None): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """
    def __init__(self, params, defaults: dict[str, Any] | None = None, **_defaults):
        if defaults is None: defaults = {}
        defaults.update(_defaults)

        super().__init__(params, defaults)
        self.global_state = self.state[self.param_groups[0]['params'][0]]
        """state of 1st parameter, can be used as global state which is how L-BFGS uses it in pytorch, and there is some kind of good reason to do it like that"""

    def get_params(self, mode: ParamFilter = 'requires_grad', cls: type[ListLike] = TensorList) -> ListLike:
        return get_params(self.param_groups, mode, cls)

    @overload
    def group_vals(self, key: str, *,
                   mode: ParamFilter = 'requires_grad', cls: type[ListLike] = NumberList) -> ListLike: ...
    @overload
    def group_vals(self, key: list[str] | tuple[str,...], *,
                   mode: ParamFilter = 'requires_grad', cls: type[ListLike] = NumberList) -> list[ListLike]: ...
    @overload
    def group_vals(self, key: str, key2: str, *keys: str,
                   mode: ParamFilter = 'requires_grad', cls: type[ListLike] = NumberList) -> list[ListLike]: ...

    def group_vals(self, key: str | list[str] | tuple[str,...], key2: str | None = None, *keys: str,
                   mode: ParamFilter = 'requires_grad', cls: type[ListLike] = NumberList) -> ListLike | list[ListLike]:
        return get_group_vals(self.param_groups, key, key2, *keys, mode = mode, cls = cls) # pyright:ignore[reportArgumentType]


    @overload
    def state_vals(self, key: str, *,
                   init: Init = torch.zeros_like,
                   mode: ParamFilter | list[torch.Tensor] | tuple[torch.Tensor, ...] = 'requires_grad',
                   cls: type[ListLike] = TensorList) -> ListLike: ...
    @overload
    def state_vals(self, key: list[str] | tuple[str,...], *,
                   init: Init | Sequence[Init] = torch.zeros_like,
                   mode: ParamFilter | list[torch.Tensor] | tuple[torch.Tensor, ...] = 'requires_grad',
                   cls: type[ListLike] = TensorList) -> list[ListLike]: ...
    @overload
    def state_vals(self, key: str, key2: str, *keys: str,
                   init: Init | Sequence[Init] = torch.zeros_like,
                   mode: ParamFilter | list[torch.Tensor] | tuple[torch.Tensor, ...] = 'requires_grad',
                   cls: type[ListLike] = TensorList) -> list[ListLike]: ...

    def state_vals(self, key: str | list[str] | tuple[str,...], key2: str | None = None, *keys: str,
                   init: Init | Sequence[Init] = torch.zeros_like,
                   mode: ParamFilter | list[torch.Tensor] | tuple[torch.Tensor, ...] = 'requires_grad',
                   cls: type[ListLike] = TensorList) -> ListLike | list[ListLike]:

        if isinstance(mode, (list,tuple)): params = mode
        else: params = self.get_params(mode)

        return get_state_vals(self.state, params, key, key2, *keys, init = init, cls = cls) # type:ignore[reportArgumentType]


    # shut up pylance
    @abstractmethod
    def step(self, closure) -> Any: ... # pylint:disable=signature-differs # pyright:ignore[reportIncompatibleMethodOverride]

def zero_grad_(params: Iterable[torch.Tensor], set_to_none):
    if set_to_none:
        for p in params:
            p.grad = None

    else:
        grads = [p.grad for p in params if p.grad is not None]
        for grad in grads:
            # taken from torch.optim.Optimizer.zero_grad
            if grad.grad_fn is not None:
                grad.detach_()
            else:
                grad.requires_grad_(False)

        torch._foreach_zero_(grads)


@overload
def unpack_states(states: Sequence[MutableMapping[str, Any]], tensors: Sequence[torch.Tensor],
                   key: str, *,
                   must_exist: bool = False, init: Init = torch.zeros_like,
                   cls: type[ListLike] = list) -> ListLike: ...
@overload
def unpack_states(states: Sequence[MutableMapping[str, Any]], tensors: Sequence[torch.Tensor],
                   key: list[str] | tuple[str,...], *,
                   must_exist: bool = False, init: Init | Sequence[Init] = torch.zeros_like,
                   cls: type[ListLike] = list) -> list[ListLike]: ...
@overload
def unpack_states(states: Sequence[MutableMapping[str, Any]], tensors: Sequence[torch.Tensor],
                   key: str,  key2: str, *keys: str,
                   must_exist: bool = False, init: Init | Sequence[Init] = torch.zeros_like,
                   cls: type[ListLike] = list) -> list[ListLike]: ...

def unpack_states(states: Sequence[MutableMapping[str, Any]], tensors: Sequence[torch.Tensor],
                   key: str | list[str] | tuple[str,...], key2: str | None = None,  *keys: str,
                   must_exist: bool = False, init: Init | Sequence[Init] = torch.zeros_like,
                   cls: type[ListLike] = list) -> ListLike | list[ListLike]:

    # single key, return single cls
    if isinstance(key, str) and key2 is None:
        values = cls()
        for i,s in enumerate(states):
            if key not in s:
                if must_exist: raise KeyError(f"Key {key} doesn't exist in state with keys {tuple(s.keys())}")
                s[key] = _make_initial_state_value(tensors[i], init, i)
            values.append(s[key])
        return values

    # multiple keys
    k1 = (key,) if isinstance(key, str) else tuple(key)
    k2 = () if key2 is None else (key2,)
    keys = k1 + k2 + keys

    values = [cls() for _ in keys]
    for i,s in enumerate(states):
        for k_i, key in enumerate(keys):
            if key not in s:
                if must_exist: raise KeyError(f"Key {key} doesn't exist in state with keys {tuple(s.keys())}")
                k_init = init[k_i] if isinstance(init, (list,tuple)) else init
                s[key] = _make_initial_state_value(tensors[i], k_init, i)
            values[k_i].append(s[key])

    return values

