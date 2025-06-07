import math
from functools import partial
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal
import warnings
import torch

from ...core import Chainable, Module, Var
from ...utils import vec_to_tensors


def _make_projected_closure(closure, var: Var, projection: "Projection",
                           params: list[torch.Tensor], projected_params: list[torch.Tensor]):

    def projected_closure(backward=True):
        unprojected_params = projection.unproject(projected_params, var, current='params')

        with torch.no_grad():
            for p, new_p in zip(params, unprojected_params):
                p.set_(new_p) # pyright: ignore[reportArgumentType]

        if backward:
            loss = closure()
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            projected_grads = projection.project(grads, var, current='grads')
            for p, g in zip(projected_params, projected_grads):
                p.grad = g

        else:
            loss = closure(False)

        return loss

    return projected_closure

def _projected_get_grad_override(
    retain_graph: bool | None = None,
    create_graph: bool = False,
    projection: Any = ...,
    unprojected_var: Any = ...,
    self: Any = ...,
):
    assert isinstance(projection, Projection)
    assert isinstance(unprojected_var, Var)
    assert isinstance(self, Var)

    if self.grad is not None: return self.grad
    grads = unprojected_var.get_grad(retain_graph, create_graph)
    projected_grads = list(projection.project(grads, self, current='grads'))
    self.grad = projected_grads
    for p, g in zip(self.params, projected_grads):
        p.grad = g
    return self.grad


class Projection(Module, ABC):
    """
    Base class for projections.
    This is an abstract class, to use it, subclass it and override `project` and `unproject`.

    Args:
        modules (Chainable): modules that will be applied in the projected domain.
        project_update (bool, optional): whether to project the update. Defaults to True.
        project_params (bool, optional):
            whether to project the params. This is necessary for modules that use closure. Defaults to False.
        project_grad (bool, optional): whether to project the gradients (separately from update). Defaults to False.
        defaults (dict[str, Any] | None, optional): dictionary with defaults. Defaults to None.
    """

    def __init__(
        self,
        modules: Chainable,
        project_update=True,
        project_params=False,
        project_grad=False,
        defaults: dict[str, Any] | None = None,
    ):
        super().__init__(defaults)
        self.set_child('modules', modules)
        self.global_state['current_step'] = 0
        self._project_update = project_update
        self._project_params = project_params
        self._project_grad = project_grad
        self._projected_params = None

    @abstractmethod
    def project(self, tensors: list[torch.Tensor], var: Var, current: Literal['params', 'grads', 'update']) -> Iterable[torch.Tensor]:
        """projects `tensors`. Note that this can be called multiple times per step with `params`, `grads`, and `update`."""

    @abstractmethod
    def unproject(self, tensors: list[torch.Tensor], var: Var, current: Literal['params', 'grads', 'update']) -> Iterable[torch.Tensor]:
        """unprojects `tensors`. Note that this can be called multiple times per step with `params`, `grads`, and `update`."""

    @torch.no_grad
    def step(self, var: Var):
        projected_var = var.clone(clone_update=False)
        update_is_grad = False

        # closure will calculate projected update and grad if needed
        if self._project_params and var.closure is not None:
            if self._project_update and var.update is not None: projected_var.update = list(self.project(var.update, var=var, current='update'))
            else:
                update_is_grad = True
            if self._project_grad and var.grad is not None: projected_var.grad = list(self.project(var.grad, var=var, current='grads'))

        # project update and grad, unprojected attributes are deleted
        else:
            if self._project_update:
                if var.update is None:
                    # update is None, meaning it will be set to `grad`.
                    # we can project grad and use it for update
                    grad = var.get_grad()
                    projected_var.grad = list(self.project(grad, var=var, current='grads'))
                    if self._project_grad: projected_var.update = [g.clone() for g in projected_var.grad]
                    else: projected_var.update = projected_var.grad.copy() # don't clone because grad shouldn't be used
                    del var.update
                    update_is_grad = True

                else:
                    update = var.get_update()
                    projected_var.update = list(self.project(update, var=var, current='update'))
                    del update, var.update

            if self._project_grad and projected_var.grad is None:
                grad = var.get_grad()
                projected_var.grad = list(self.project(grad, var=var, current='grads'))

        original_params = None
        if self._project_params:
            original_params = [p.clone() for p in var.params]
            projected_params = self.project(var.params, var=var, current='params')

        else:
            # make fake params for correct shapes and state storage
            # they reuse update or grad storage for memory efficiency
            projected_params = projected_var.update if projected_var.update is not None else projected_var.grad
            assert projected_params is not None

        if self._projected_params is None:
            # 1st step - create objects for projected_params. They have to remain the same python objects
            # to support per-parameter states which are stored by ids.
            self._projected_params = [p.view_as(p).requires_grad_() for p in projected_params]
        else:
            # set storage to new fake params while ID remains the same
            for empty_p, new_p in zip(self._projected_params, projected_params):
                empty_p.set_(new_p.view_as(new_p).requires_grad_()) # pyright: ignore[reportArgumentType]

        # project closure
        if self._project_params:
            closure = var.closure; params = var.params
            projected_var.closure = _make_projected_closure(closure, var=var, projection=self, params=params,
                                                             projected_params=self._projected_params)

        else:
            projected_var.closure = None

        # step
        projected_var.params = self._projected_params
        projected_var.get_grad = partial(
            _projected_get_grad_override,
            projection=self,
            unprojected_var=var,
            self=projected_var,
        )
        projected_var = self.children['modules'].step(projected_var)

        # empty fake params storage
        # this doesn't affect update/grad because it is a different python object, set_ changes storage on an object
        if not self._project_params:
            for p in self._projected_params:
                p.set_(torch.empty(0, device=p.device, dtype=p.dtype)) # pyright: ignore[reportArgumentType]

        # unproject
        unprojected_var = projected_var.clone(clone_update=False)
        unprojected_var.closure = var.closure
        unprojected_var.params = var.params
        unprojected_var.grad = var.grad

        if self._project_update:
            assert projected_var.update is not None
            unprojected_var.update = list(self.unproject(projected_var.update, var=var, current='grads' if update_is_grad else 'update'))
            del projected_var.update

        # unprojecting grad doesn't make sense?
        # if self._project_grad:
        #     assert projected_var.grad is not None
        #     unprojected_var.grad = list(self.unproject(projected_var.grad, var=var))

        del projected_var

        if original_params is not None:
            for p, o in zip(unprojected_var.params, original_params):
                p.set_(o) # pyright: ignore[reportArgumentType]

        return unprojected_var



class FlipConcatProjection(Projection):
    """
    for testing
    """

    def __init__(self, modules: Chainable, project_update=True, project_params=False, project_grad=False):
        super().__init__(modules, project_update=project_update, project_params=project_params, project_grad=project_grad)

    @torch.no_grad
    def project(self, tensors, var, current):
        return [torch.cat([u.view(-1) for u in tensors], dim=-1).flip(0)]

    @torch.no_grad
    def unproject(self, tensors, var, current):
        return vec_to_tensors(vec=tensors[0].flip(0), reference=var.params)


class NoopProjection(Projection):
    """an example projection which doesn't do anything for testing"""

    def __init__(self, modules: Chainable, project_update=True, project_params=False, project_grad=False):
        super().__init__(modules, project_update=project_update, project_params=project_params, project_grad=project_grad)

    @torch.no_grad
    def project(self, tensors, var, current):
        return tensors

    @torch.no_grad
    def unproject(self, tensors, var, current):
        return tensors

class MultipyProjection(Projection):
    """an example projection which multiplies everything by 2"""

    def __init__(self, modules: Chainable, project_update=True, project_params=False, project_grad=False):
        super().__init__(modules, project_update=project_update, project_params=project_params, project_grad=project_grad)

    @torch.no_grad
    def project(self, tensors, var, current):
        return torch._foreach_mul(tensors, 2)

    @torch.no_grad
    def unproject(self, tensors, var, current):
        return torch._foreach_div(tensors, 2)

