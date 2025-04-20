from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal

import torch

from ...core import Chainable, Module, Vars
from ...utils import vec_to_tensors


def _make_projected_closure(closure, vars: Vars, projection: "Projection",
                           params: list[torch.Tensor], projected_params: list[torch.Tensor]):

    def projected_closure(backward=True):
        unprojected_params = projection.unproject(projected_params, vars)

        with torch.no_grad():
            for p, new_p in zip(params, unprojected_params):
                p.set_(new_p) # pyright: ignore[reportArgumentType]

        if backward:
            loss = closure()
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            projected_grads = projection.project(grads, vars)
            for p, g in zip(projected_params, projected_grads):
                p.grad = g

        else:
            loss = closure(False)

        return loss

    return projected_closure


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
    def project(self, tensors: list[torch.Tensor], vars: Vars) -> Iterable[torch.Tensor]:
        """projects `tensors`."""

    @abstractmethod
    def unproject(self, tensors: list[torch.Tensor], vars: Vars) -> Iterable[torch.Tensor]:
        """unprojects `tensors` after projecting and updating them"""

    @torch.no_grad
    def step(self, vars: Vars):
        projected_vars = vars.clone(clone_update=False)

        # closure will calculate projected update and grad if needed
        if self._project_params and vars.closure is not None:
            if self._project_update and vars.update is not None: projected_vars.update = list(self.project(vars.update, vars=vars))
            if self._project_grad and vars.grad is not None: projected_vars.grad = list(self.project(vars.grad, vars=vars))

        # project update and grad, unprojected attributes are deleted
        else:
            if self._project_update:
                if vars.update is None:
                    # update is None, meaning it will be set to `grad`.
                    # we can project grad and use it for update
                    grad = vars.get_grad()
                    projected_vars.grad = list(self.project(grad, vars=vars))
                    if self._project_grad: projected_vars.update = [g.clone() for g in projected_vars.grad]
                    else: projected_vars.update = projected_vars.grad.copy() # don't clone because grad shouldn't be used
                    del vars.update

                else:
                    update = vars.get_update()
                    projected_vars.update = list(self.project(update, vars=vars))
                    del update, vars.update

            if self._project_grad and projected_vars.grad is None:
                grad = vars.get_grad()
                projected_vars.grad = list(self.project(grad, vars=vars))

        original_params = None
        if self._project_params:
            original_params = [p.clone() for p in vars.params]
            projected_params = self.project(vars.params, vars=vars)

        else:
            # make fake params for correct shapes and state storage
            # they reuse update or grad storage for memory efficiency
            projected_params = projected_vars.update if projected_vars.update is not None else projected_vars.grad
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
            closure = vars.closure; params = vars.params
            projected_vars.closure = _make_projected_closure(closure, vars=vars, projection=self, params=params,
                                                             projected_params=self._projected_params)

        else:
            projected_vars.closure = None

        # step
        projected_vars.params = self._projected_params
        projected_vars = self.children['modules'].step(projected_vars)

        # empty fake params storage
        # this doesn't affect update/grad because it is a different python object, set_ changes storage on an object
        if not self._project_params:
            for p in self._projected_params:
                p.set_(torch.empty(0, device=p.device, dtype=p.dtype)) # pyright: ignore[reportArgumentType]

        # unproject
        unprojected_vars = projected_vars.clone(clone_update=False)
        unprojected_vars.closure = vars.closure
        unprojected_vars.params = vars.params
        if unprojected_vars.grad is None: unprojected_vars.grad = vars.grad

        if self._project_update:
            assert projected_vars.update is not None
            unprojected_vars.update = list(self.unproject(projected_vars.update, vars=vars))
            del projected_vars.update

        # unprojecting grad doesn't make sense?
        # if self._project_grad:
        #     assert projected_vars.grad is not None
        #     unprojected_vars.grad = list(self.unproject(projected_vars.grad, vars=vars))

        del projected_vars

        if original_params is not None:
            for p, o in zip(unprojected_vars.params, original_params):
                p.set_(o) # pyright: ignore[reportArgumentType]

        return unprojected_vars



class ConcatProjection(Projection):
    """
    an example projection, where all parameters are concatenated into a single vector which is optimized as single parameter.

    This is used for tests and it should not change the optimizer behavior in most cases (unless norm/sum/mean is used).

    This can also make some modules faster, for example LBFGS, due to operating on single vectors.
    """

    def __init__(self, modules: Chainable, project_update=True, project_params=False, project_grad=False):
        super().__init__(modules, project_update=project_update, project_params=project_params, project_grad=project_grad)

    @torch.no_grad
    def project(self, tensors, vars):
        return [torch.cat([u.view(-1) for u in tensors], dim=-1)]

    @torch.no_grad
    def unproject(self, tensors, vars):
        return vec_to_tensors(vec=tensors[0], reference=vars.params)


class FlipConcatProjection(Projection):
    """
    for testing
    """

    def __init__(self, modules: Chainable, project_update=True, project_params=False, project_grad=False):
        super().__init__(modules, project_update=project_update, project_params=project_params, project_grad=project_grad)

    @torch.no_grad
    def project(self, tensors, vars):
        return [torch.cat([u.view(-1) for u in tensors], dim=-1).flip(0)]

    @torch.no_grad
    def unproject(self, tensors, vars):
        return vec_to_tensors(vec=tensors[0].flip(0), reference=vars.params)


class NoopProjection(Projection):
    """an example projection which doesn't do anything for testing"""

    def __init__(self, modules: Chainable, project_update=True, project_params=False, project_grad=False):
        super().__init__(modules, project_update=project_update, project_params=project_params, project_grad=project_grad)

    @torch.no_grad
    def project(self, tensors, vars):
        return tensors

    @torch.no_grad
    def unproject(self, tensors, vars):
        return

class MultipyProjection(Projection):
    """an example projection which multiplies everything by 2"""

    def __init__(self, modules: Chainable, project_update=True, project_params=False, project_grad=False):
        super().__init__(modules, project_update=project_update, project_params=project_params, project_grad=project_grad)

    @torch.no_grad
    def project(self, tensors, vars):
        return torch._foreach_mul(tensors, 2)

    @torch.no_grad
    def unproject(self, tensors, vars):
        return torch._foreach_div(tensors, 2)