
import warnings
from abc import ABC, abstractmethod
from collections import ChainMap, defaultdict
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from operator import itemgetter
from typing import Any, final, overload, Literal, cast, TYPE_CHECKING

import torch

from ..utils import (
    Init,
    ListLike,
    Params,
    _make_param_groups,
    get_state_vals,
    vec_to_tensors
)
from ..utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward, flatten_jacobian
from ..utils.python_tools import flatten
from ..utils.linalg.linear_operator import LinearOperator

if TYPE_CHECKING:
    from .modular import Modular

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
    Holds parameters, gradient, update, objective function (closure) if supplied, loss, and some other info.
    Modules take in a ``Var`` object, modify and it is passed to the next module.

    """
    def __init__(
        self,
        params: list[torch.Tensor],
        closure: Callable | None,
        model: torch.nn.Module | None,
        current_step: int,
        parent: "Var | None" = None,
        modular: "Modular | None" = None,
        loss: torch.Tensor | None = None,
        storage: dict | None = None,
    ):
        self.params: list[torch.Tensor] = params
        """List of all parameters with requires_grad = True."""

        self.closure = closure
        """A closure that reevaluates the model and returns the loss, None if it wasn't specified"""

        self.model = model
        """torch.nn.Module object of the model, None if it wasn't specified."""

        self.current_step: int = current_step
        """global current step, starts at 0. This may not correspond to module current step,
        for example a module may step every 10 global steps."""

        self.parent: "Var | None" = parent
        """parent ``Var`` object. When ``self.get_grad()`` is called, it will also set ``parent.grad``.
        Same with ``self.get_loss()``. This is useful when ``self.params`` are different from ``parent.params``,
        e.g. when projecting."""

        self.modular: "Modular | None" = modular
        """Modular optimizer object that created this ``Var``."""

        self.update: list[torch.Tensor] | None = None
        """
        current update. Update is assumed to be a transformed gradient, therefore it is subtracted.

        If closure is None, this is initially set to cloned gradient. Otherwise this is set to None.

        At the end ``var.get_update()`` is subtracted from parameters. Therefore if ``var.update`` is ``None``,
        gradient will be used and calculated if needed.
        """

        self.grad: list[torch.Tensor] | None = None
        """gradient with current parameters. If closure is not ``None``, this is set to ``None`` and can be calculated if needed."""

        self.loss: torch.Tensor | Any | None = loss
        """loss with current parameters."""

        self.loss_approx: torch.Tensor | Any | None = None
        """loss at a point near current point. This can be useful as some modules only calculate loss at perturbed points,
        whereas some other modules require loss strictly at current point."""

        self.post_step_hooks: list[Callable[[Modular, Var]]] = []
        """list of functions to be called after optimizer step.

        This attribute should always be modified in-place (using ``append`` or ``extend``).

        The signature is:

        ```python
        def hook(optimizer: Modular, var: Vars): ...
        ```
        """

        self.stop: bool = False
        """if True, all following modules will be skipped.
        If this module is a child, it only affects modules at the same level (in the same Chain)."""

        self.skip_update: bool = False
        """if True, the parameters will not be updated."""

        # self.storage: dict = {}
        # """Storage for any other data, such as hessian estimates, etc."""

        self.attrs: dict = {}
        """attributes, Modular.attrs is updated with this after each step. This attribute should always be modified in-place"""

        if storage is None: storage = {}
        self.storage: dict = storage
        """additional kwargs passed to closure will end up in this dict. This attribute should always be modified in-place"""

        self.should_terminate: bool | None = None
        """termination criteria, Modular.should_terminate is set to this after each step if not None"""

    def get_loss(self, backward: bool, retain_graph = None, create_graph: bool = False) -> torch.Tensor | float:
        """Returns the loss at current parameters, computing it if it hasn't been computed already and assigning ``var.loss``.
        Do not call this at perturbed parameters. Backward always sets grads to None before recomputing."""
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

                # next time closure() is called, it will set grad to None.
                # zero_grad(set_to_none=False) shouldn't be used (I should add a warning)
                self.grad = [p.grad if p.grad  is not None else torch.zeros_like(p) for p in self.params]
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

        # set parent grad
        if self.parent is not None:
            # the way projections/split work, they make a new closure which evaluates original
            # closure and projects the gradient, and set it as their var.closure.
            # then on `get_loss(backward=True)` it is called, so it also sets original parameters gradient.
            # and we set it to parent var here.
            if self.parent.loss is None: self.parent.loss = self.loss
            if self.parent.grad is None and backward:
                if all(p.grad is None for p in self.parent.params):
                    warnings.warn("Parent grad is None after backward.")
                self.parent.grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.parent.params]

        return self.loss # type:ignore

    def get_grad(self, retain_graph: bool | None = None, create_graph: bool = False) -> list[torch.Tensor]:
        """Returns the gradient at initial parameters, computing it if it hasn't been computed already and assigning
        ``var.grad`` and potentially ``var.loss``. Do not call this at perturbed parameters."""
        if self.grad is None:
            if self.closure is None: raise RuntimeError("closure is None")
            self.get_loss(backward=True, retain_graph=retain_graph, create_graph=create_graph) # evaluate and set self.loss and self.grad

        assert self.grad is not None
        return self.grad

    def get_update(self) -> list[torch.Tensor]:
        """Returns the update. If update is None, it is initialized by cloning the gradients and assigning to ``var.update``.
        Computing the gradients may assign ``var.grad`` and ``var.loss`` if they haven't been computed.
        Do not call this at perturbed parameters."""
        if self.update is None: self.update = [g.clone() for g in self.get_grad()]
        return self.update

    def clone(self, clone_update: bool, parent: "Var | None" = None):
        """Creates a shallow copy of the Vars object, update can optionally be deep-copied (via ``torch.clone``).

        Setting ``parent`` is only if clone's parameters are something different,
        while clone's closure referes to the same objective but with a "view" on parameters.
        """
        copy = Var(params = self.params, closure=self.closure, model=self.model, current_step=self.current_step, parent=parent)

        if clone_update and self.update is not None:
            copy.update = [u.clone() for u in self.update]
        else:
            copy.update = self.update

        copy.grad = self.grad
        copy.loss = self.loss
        copy.loss_approx = self.loss_approx
        copy.closure = self.closure
        copy.post_step_hooks = self.post_step_hooks
        copy.stop = self.stop
        copy.skip_update = self.skip_update

        copy.modular = self.modular
        copy.attrs = self.attrs
        copy.storage = self.storage
        copy.should_terminate = self.should_terminate

        return copy

    def update_attrs_from_clone_(self, var: "Var"):
        """Updates attributes of this `Vars` instance from a cloned instance.
        Typically called after a child module has processed a cloned `Vars`
        object. This propagates any newly computed loss or gradient values
        from the child's context back to the parent `Vars` if the parent
        didn't have them computed already.

        Also, as long as ``post_step_hooks`` and ``attrs`` are modified in-place,
        if the child updates them, the update will affect the parent too.
        """
        if self.loss is None: self.loss = var.loss
        if self.loss_approx is None: self.loss_approx = var.loss_approx
        if self.grad is None: self.grad = var.grad

        if var.should_terminate is not None: self.should_terminate = var.should_terminate

    def zero_grad(self, set_to_none=True):
        if set_to_none:
            for p in self.params: p.grad = None
        else:
            grads = [p.grad for p in self.params if p.grad is not None]
            if len(grads) != 0: torch._foreach_zero_(grads)

# endregion
