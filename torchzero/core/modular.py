
import warnings
from abc import ABC, abstractmethod
from collections import ChainMap, defaultdict
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Literal, cast, final, overload

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
from .module import Chainable, Module
from .var import Var
from .functional import step

class _EvalCounterClosure:
    """keeps track of how many times closure has been evaluated, and sets closure return"""
    __slots__ = ("modular", "closure")
    def __init__(self, modular: "Modular", closure):
        self.modular = modular
        self.closure = closure

    def __call__(self, *args, **kwargs):
        if self.closure is None:
            raise RuntimeError("One of the modules requires closure to be passed to the step method")

        v = self.closure(*args, **kwargs)

        # set closure return on 1st evaluation
        if self.modular._closure_return is None:
            self.modular._closure_return = v

        self.modular.num_evaluations += 1
        return v


def unroll_modules(*modules: Chainable) -> list[Module]:
    unrolled = []

    for m in modules:
        if isinstance(m, Module):
            unrolled.append(m)
            unrolled.extend(unroll_modules(list(m.children.values())))
        else:
            unrolled.extend(unroll_modules(*m))

    return unrolled


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
        if len(modules) == 0: raise RuntimeError("Empty list of modules passed to `Modular`")
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
        """Maps each parameter tensor to a list of per-module global settings.
        Each element in the list is ChainDict's 2nd map of a module."""

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

        # note - this is what super().__init__(param_groups, defaults=defaults) does:

        # self.defaults = defaults
        # for param_group in param_groups:
        #     self.add_param_group(param_group)

        # add_param_group adds a ChainMap where defaults are lowest priority,
        # and entries specifed in param_groups or scheduler are higher priority.
        # pytorch schedulers do group["lr"] = new_lr, which sets higher priority key.
        # in each module, settings passed to that module by calling set_param_groups are highest priority

        self.current_step = 0
        """global step counter for the optimizer."""

        self.num_evaluations = 0
        """number of times the objective has been evaluated (number of closure calls or number of steps if closure is None)."""

        # reformulations will change the closure to return a different loss (e.g. a sqrt homotopy, gaussian homotopy)
        # we want to return original loss so this attribute is used
        self._closure_return = None
        """on each step, first time a closure is evaluated, this attribute is set to the returned value. `step` method returns this."""

        self.attrs = {}
        """custom attributes that can be set by modules, for example EMA of weights or best so far"""

        self.should_terminate = False
        """is set to True by termination criteria modules."""

    def add_param_group(self, param_group: dict[str, Any]):
        proc_param_group = _make_param_groups([param_group], differentiable=False)[0]
        self.param_groups.append(ChainMap(proc_param_group, self.defaults))
        # setting param_group[key] = value sets it to first map (the `proc_param_group`).
        # therefore lr schedulers override defaults, but not settings passed to individual modules
        # by `set_param_groups` .

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
            m._load_state_dict(sd, id_to_tensor)


    def step(self, closure=None, loss=None, **kwargs): # pyright: ignore[reportIncompatibleMethodOverride]
        # clear closure return from previous step
        self._closure_return = None

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
        var = Var(params=params, closure=_EvalCounterClosure(self, closure), model=self.model, current_step=self.current_step, modular=self, loss=loss, storage=kwargs)

        # if closure is None, assume backward has been called and gather grads
        if closure is None:
            var.grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            self.num_evaluations += 1

        if len(self.modules) == 0: raise RuntimeError("There are no modules in this `Modular` optimizer")

        # step
        var = step(var, self.modules)

        # apply update
        if not var.skip_update:
            with torch.no_grad():
                torch._foreach_sub_(params, var.get_update())

        # update attributes
        self.attrs.update(var.attrs)
        if var.should_terminate is not None: self.should_terminate = var.should_terminate

        # hooks
        for hook in var.post_step_hooks:
            hook(self, var)

        self.current_step += 1
        #return var.loss if var.loss is not None else var.loss_approx
        return self._closure_return

    def __repr__(self):
        return f'Modular({", ".join(str(m) for m in self.modules)})'

