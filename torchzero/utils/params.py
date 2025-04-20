from typing import Any
from collections.abc import Sequence, Iterable, Mapping
import warnings
import torch, numpy as np



Params = Iterable[torch.Tensor | tuple[str, torch.Tensor] | Mapping[str, Any]]

def _validate_params_are_unique_(params: Sequence[torch.Tensor]):
    # this is from pytorch add_param_group
    if len(params) != len(set(params)):
        warnings.warn(
            "optimizer contains a parameter group with duplicate parameters; "
            "in future, this will cause an error; "
            "see github.com/pytorch/pytorch/issues/40967 for more information",
            stacklevel=3,
        )

def _validate_param_is_differentiable_(tensor: torch.Tensor | Any):
    """Checks that param is torch.Tensor and isn't a leaf parameter unless differentiable is True, otherwise this raises, this is taken from torch.optim.Optimizer."""
    if not (tensor.is_leaf or tensor.retains_grad):
        raise ValueError("can't optimize a non-leaf Tensor")

def _validate_at_least_one_param_requires_grad_(params: Iterable[torch.Tensor]):
    params = list(params)
    if not any(p.requires_grad for p in params):
        warnings.warn(
            "Parameter group contains no parameters which require gradients. "
            "Note for gradient-free optimizers, they still only optimize parameters with requires_grad=True, "
            "so if needed, use `with torch.no_grad():` context instead.", stacklevel=3)



def _copy_param_groups(param_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """copies param_groups, doesn't copy the tensors."""
    new_param_group = []

    for g in param_groups:
        assert isinstance(g, dict)
        g_copy = g.copy()

        for k in ('params', 'updates', 'grads'):
            if k in g_copy:
                assert isinstance(g_copy[k], list)
                g_copy[k] = g_copy[k].copy()

        new_param_group.append(g_copy)

    return new_param_group

def _process_param_group_(param_group: dict[str, Any]) -> dict[str, Any]:
    """makes sure `param_group["params"]` is a list of tensors, and sets `param_group["param_names"]` if params are named."""
    if 'params' not in param_group: raise KeyError("Param group doesn't have a `params` key.")

    if isinstance(param_group['params'], torch.Tensor): param_group['params'] = [param_group['params']]

    tensors: list[torch.Tensor] = []
    names: list[str] | None = []

    for p in param_group['params']:
        if isinstance(p, torch.Tensor):
            tensors.append(p)

        elif isinstance(p, tuple):
            if len(p) != 2:
                raise ValueError(f'named_parameters must be a tuple of (name, tensor), got length {len(p)} tuple')
            if (not isinstance(p[0], str)) or (not isinstance(p[1], torch.Tensor)):
                raise ValueError(f'named_parameters must be a tuple of (name, tensor), got {[type(a) for a in p]}')
            names.append(p[0])
            tensors.append(p[1])

        else:
            raise ValueError(f'Parameters must be tensors or tuples (name, tensor), got parameter of type {type(p)}')

    if len(tensors) == 0: warnings.warn('got an empty parameter group')

    param_group['params'] = tensors

    if len(names) != 0:
        if len(names) != len(tensors):
            raise ValueError(f"Number of parameters {len(tensors)} doesn't match number of names {len(names)}")
        param_group['param_names'] = names

    return param_group

def _make_param_groups(params: Params, differentiable: bool) -> list[dict[str, Any]]:
    params = list(params)

    param_groups: list[dict[str, Any]] = [dict(p) for p in params if isinstance(p, Mapping)]
    tensors = [p for p in params if isinstance(p, torch.Tensor)]
    named_tensors = [p for p in params if isinstance(p, tuple)]

    if len(tensors) != 0: param_groups.append({"params": tensors})
    if len(named_tensors) != 0: param_groups.append({"params": named_tensors})

    # process param_groups
    for g in param_groups:
        _process_param_group_(g)

    # validate
    all_params = [p for g in param_groups for p in g['params']]
    _validate_params_are_unique_(all_params)
    _validate_at_least_one_param_requires_grad_(all_params)
    if not differentiable:
        for p in all_params: _validate_param_is_differentiable_(p)

    return param_groups

def _add_defaults_to_param_groups_(param_groups: list[dict[str, Any]], defaults: dict[str, Any]) -> list[dict[str, Any]]:
    for group in param_groups:
        for k, v in defaults.items():
            if k not in group:
                group[k] = v
    return param_groups

def _add_updates_grads_to_param_groups_(param_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for group in param_groups:
        if 'updates' in group: raise ValueError('updates in group')
        group['updates'] = [None for _ in group['params']]

        if 'grads' in group: raise ValueError('grads in group')
        group['grads'] = [None for _ in group['grads']]

    return param_groups


def _set_update_and_grad_(
    param_groups: list[dict[str, Any]],
    updates: list[torch.Tensor] | None,
    grads: list[torch.Tensor] | None,
) -> list[dict[str, Any]]:
    if updates is None and grads is None: return param_groups

    updates_iter = iter(updates) if updates is not None else None
    grads_iter = iter(grads) if grads is not None else None

    for group in param_groups:
        group_params = group['params']
        group_updates = group['updates']
        group_grads = group['grads']

        for i, param in enumerate(group_params):
            if not param.requires_grad: continue
            if updates_iter is not None: group_updates[i] = next(updates_iter)
            if grads_iter is not None: group_grads[i] = next(grads_iter)

    return param_groups

