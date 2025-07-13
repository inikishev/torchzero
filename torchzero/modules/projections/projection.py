import math
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, ChainMap
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Literal

import torch

from ...core import Chainable, Module, Var
from ...utils import vec_to_tensors, set_storage_


def _make_projected_closure(closure, project_fn, unproject_fn,
                           params: list[torch.Tensor], projected_params: list[torch.Tensor]):
    def projected_closure(backward=True):
        # unproject projected params
        unprojected_params = unproject_fn(projected_tensors=projected_params, current='params')

        # set actual model parameters to suggested parameters
        with torch.no_grad():
            for p, new_p in zip(params, unprojected_params):
                p.set_(new_p) # pyright: ignore[reportArgumentType]

        # evaluate closure with suggested parameters
        if backward:
            loss = closure()
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

            # project gradients on backward and set to projected parameter .grad attributes
            projected_grads = project_fn(grads, current='grads')
            for p, g in zip(projected_params, projected_grads):
                p.grad = g

        else:
            loss = closure(False)

        return loss

    return projected_closure

class _FakeProjectedClosure:
    """This is used when project_params is False. Then the closure is meant to only be used to evaluate the initial gradient.
    It should just evaluate original closure, project the gradients, and set them to fake params.

    I made it into a class so that it can know and raise when it evaluates closure more than once.
    """
    __slots__ = ('closure', 'project_fn', 'params', 'fake_params', 'evaluated')
    def __init__(self, closure, project_fn, params: list[torch.Tensor], fake_params: list[torch.Tensor]):
        self.closure = closure
        self.project_fn = project_fn
        self.params = params
        self.fake_params = fake_params
        self.evaluated = False

    def __call__(self, backward: bool = True):
        if self.evaluated:
            raise RuntimeError("set project_params to True if projected modules require closure.")
        self.evaluated = True

        # evaluate closure with suggested parameters
        if backward:

            loss = self.closure()
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]

            # project gradients on backward and set to projected parameter .grad attributes
            projected_grads = self.project_fn(grads, current='grads')
            for p, g in zip(self.fake_params, projected_grads):
                p.grad = g

        else:
            loss = self.closure(False)

        return loss



class ProjectionBase(Module, ABC):
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

        self._states: dict[str, list[dict[str, Any]]] = {}
        """per-parameter states for each projection target"""

    @abstractmethod
    def project(
        self,
        tensors: list[torch.Tensor],
        params: list[torch.Tensor],
        grads: list[torch.Tensor] | None,
        loss: torch.Tensor | None,
        states: list[dict[str, Any]],
        settings: list[ChainMap[str, Any]],
        current: str,
    ) -> Iterable[torch.Tensor]:
        """projects `tensors`. Note that this can be called multiple times per step with `params`, `grads`, and `update`."""

    @abstractmethod
    def unproject(
        self,
        projected_tensors: list[torch.Tensor],
        params: list[torch.Tensor],
        grads: list[torch.Tensor] | None,
        loss: torch.Tensor | None,
        states: list[dict[str, Any]],
        settings: list[ChainMap[str, Any]],
        current: str,
    ) -> Iterable[torch.Tensor]:
        """unprojects `tensors`. Note that this can be called multiple times per step with `params`, `grads`, and `update`.

        Args:
            projected_tensors (list[torch.Tensor]): projected tensors to unproject.
            params (list[torch.Tensor]): original, unprojected parameters.
            grads (list[torch.Tensor] | None): original, unprojected gradients
            loss (torch.Tensor | None): loss at initial point.
            states (list[dict[str, Any]]): list of state dictionaries per each UNPROJECTED tensor.
            settings (list[ChainMap[str, Any]]): list of setting dictionaries per each UNPROJECTED tensor.
            current (str): string representing what is being unprojected, e.g. "params", "grads" or "update".

        Returns:
            Iterable[torch.Tensor]: unprojected tensors of the same shape as params
        """

    @torch.no_grad
    def step(self, var: Var):
        params = var.params
        settings = [self.settings[p] for p in params]

        def _project(tensors: list[torch.Tensor], current: Literal['params', 'grads', 'update']):
            states = self._states.setdefault(current, [{} for _ in params])
            return list(self.project(
                tensors=tensors,
                params=params,
                grads=var.grad,
                loss=var.loss,
                states=states,
                settings=settings,
                current=current,
            ))

        projected_var = var.clone(clone_update=False)

        closure = var.closure

        # if this is True, update and grad were projected simultaneously under current="grads"
        # so update will have to be unprojected with current="grads"
        update_is_grad = False

        # if closure is provided and project_params=True, make new closure that evaluates projected params
        # that also means projected modules can evaluate grad/update at will, it shouldn't be computed here
        # but if it has already been computed, it should be projected
        if self._project_params and closure is not None:

            if self._project_update and var.update is not None:
                # project update only if it already exists
                projected_var.update = _project(var.update, current='update')

            else:
                # update will be set to gradients on var.get_grad()
                # therefore projection will happen with current="grads"
                update_is_grad = True

            # project grad only if it already exists
            if self._project_grad and var.grad is not None:
                projected_var.grad = _project(var.grad, current='grads')

        # otherwise update/grad needs to be calculated and projected here
        else:
            if self._project_update:
                if var.update is None:
                    # update is None, meaning it will be set to `grad`.
                    # we can project grad and use it for update
                    grad = var.get_grad()
                    projected_var.grad = _project(grad, current='grads')
                    projected_var.update = [g.clone() for g in projected_var.grad]
                    del var.update
                    update_is_grad = True

                else:
                    # update exists so it needs to be projected
                    update = var.get_update()
                    projected_var.update = _project(update, current='update')
                    del update, var.update

            if self._project_grad and projected_var.grad is None:
                # projected_vars.grad may have been projected simultaneously with update
                # but if that didn't happen, it is projected here
                grad = var.get_grad()
                projected_var.grad = _project(grad, current='grads')


        original_params = None
        if self._project_params:
            original_params = [p.clone() for p in var.params]
            projected_params = _project(var.params, current='params')

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

        projected_params = self._projected_params
        # projected_settings = [self.settings[p] for p in projected_params]

        def _unproject(projected_tensors: list[torch.Tensor], current: Literal['params', 'grads', 'update']):
            states = self._states.setdefault(current, [{} for _ in params])
            return list(self.unproject(
                projected_tensors=projected_tensors,
                params=params,
                grads=var.grad,
                loss=var.loss,
                states=states,
                settings=settings,
                current=current,
            ))

        # project closure
        if self._project_params:
            projected_var.closure = _make_projected_closure(closure, project_fn=_project, unproject_fn=_unproject,
                                                            params=params, projected_params=projected_params)

        elif closure is not None:
            projected_var.closure = _FakeProjectedClosure(closure, project_fn=_project,
                                                          params=params, fake_params=projected_params)

        else:
            projected_var.closure = None

        # ----------------------------------- step ----------------------------------- #
        projected_var.params = projected_params
        projected_var = self.children['modules'].step(projected_var)

        # empty fake params storage
        # this doesn't affect update/grad because it is a different python object, set_ changes storage on an object
        if not self._project_params:
            for p in self._projected_params:
                set_storage_(p, torch.empty(0, device=p.device, dtype=p.dtype))

        # --------------------------------- unproject -------------------------------- #
        unprojected_var = projected_var.clone(clone_update=False)
        unprojected_var.closure = var.closure
        unprojected_var.params = var.params
        unprojected_var.grad = var.grad

        if self._project_update:
            assert projected_var.update is not None
            unprojected_var.update = _unproject(projected_var.update, current='grads' if update_is_grad else 'update')
            del projected_var.update

        del projected_var

        # original params are stored if params are projected
        if original_params is not None:
            for p, o in zip(unprojected_var.params, original_params):
                p.set_(o) # pyright: ignore[reportArgumentType]

        return unprojected_var



# basic examples
class VectorProjection(ProjectionBase):
    """projection that concatenates all parameters into a vector"""
    def __init__(
        self,
        modules: Chainable,
        project_update=True,
        project_params=True,
        project_grad=True,
    ):
        super().__init__(modules, project_update=project_update, project_params=project_params, project_grad=project_grad)

    @torch.no_grad
    def project(self, tensors, params, grads, loss, states, settings, current):
        return [torch.cat([t.ravel() for t in tensors])]

    @torch.no_grad
    def unproject(self, projected_tensors, params, grads, loss, states, settings, current):
        return vec_to_tensors(vec=projected_tensors[0], reference=params)


class ScalarProjection(ProjectionBase):
    """projetion that splits all parameters into individual scalars"""
    def __init__(
        self,
        modules: Chainable,
        project_update=True,
        project_params=True,
        project_grad=True,
    ):
        super().__init__(modules, project_update=project_update, project_params=project_params, project_grad=project_grad)

    @torch.no_grad
    def project(self, tensors, params, grads, loss, states, settings, current):
        return [s for t in tensors for s in t.ravel().unbind(0)]

    @torch.no_grad
    def unproject(self, projected_tensors, params, grads, loss, states, settings, current):
        return vec_to_tensors(vec=torch.stack(projected_tensors), reference=params)

