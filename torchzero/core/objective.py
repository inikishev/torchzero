
import warnings
from collections.abc import Callable, Sequence, Iterable
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from ..utils import Distributions, TensorList, vec_to_tensors, set_storage_
from ..utils.derivatives import (
    flatten_jacobian,
    hessian_mat,
    hvp_fd_central,
    hvp_fd_forward,
    jacobian_and_hessian_wrt,
    jacobian_wrt,
    hessian_fd,
)
from ..utils.thoad_tools import thoad_derivatives, thoad_single_tensor, lazy_thoad

if TYPE_CHECKING:
    from .modular import Optimizer
    from .module import Module

def _closure_backward(closure, params, backward, retain_graph, create_graph):
    """Calls closure with specified ``backward``, ``retain_graph`` and ``create_graph``.

    Returns loss and sets ``param.grad`` attributes.

    If ``backward=True``, this uses ``torch.enable_grad()`` context.
    """
    if not backward:
        return closure(False)

    with torch.enable_grad():
        if not (retain_graph or create_graph):
            return closure()

        # zero grad (because closure called with backward=False)
        for p in params: p.grad = None

        # loss
        loss = closure(False).ravel()

        # grad
        grad = torch.autograd.grad(
            loss,
            params,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=True,
            materialize_grads=True,
    )

    # set p.grad
    for p,g in zip(params,grad): p.grad = g
    return loss

@torch.enable_grad
def _closure_loss_grad(closure, params, retain_graph, create_graph) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Calls closure with specified ``backward``, ``retain_graph`` and ``create_graph``
    within ``torch.enable_grad()``context.

    Returns ``(loss, grad)``. Unlike ``_closure_backward``, this won't always set ``p.grad``.
    """
    if closure is None: raise RuntimeError("closure is None")

    # use torch.autograd.grad
    if retain_graph or create_graph:
        loss = closure(False).ravel()
        return loss, list(
            torch.autograd.grad(loss, params, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True, materialize_grads=True)
        )

    # use backward
    loss = closure()
    return loss, [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

HVPMethod = Literal["batched_autograd", "autograd", "fd_forward", "fd_central"]
"""
Determines how hessian-vector products are computed.

- ``"batched_autograd"`` - uses autograd with batched hessian-vector products. If a single hessian-vector is evaluated, equivalent to ``"autograd"``. Faster than ``"autograd"`` but uses more memory.
- ``"autograd"`` - uses autograd hessian-vector products. If multiple hessian-vector products are evaluated, uses a for-loop. Slower than ``"batched_autograd"`` but uses less memory.
- ``"fd_forward"`` - uses gradient finite difference approximation with a less accurate forward formula which requires one extra gradient evaluation per hessian-vector product.
- ``"fd_central"`` - uses gradient finite difference approximation with a more accurate central formula which requires two gradient evaluations per hessian-vector product.

Defaults to ``"autograd"``.
"""

HessianMethod = Literal[
    "batched_autograd",
    "autograd",
    "functional_revrev",
    "functional_fwdrev",
    "func",
    "gfd_forward",
    "gfd_central",
    "fd",
    "fd_full",
    "thoad",
]
"""
Determines how hessian is computed.

- ``"batched_autograd"`` - uses autograd to compute ``ndim`` batched hessian-vector products. Faster than ``"autograd"`` but uses more memory.
- ``"autograd"`` - uses autograd to compute ``ndim`` hessian-vector products using for loop. Slower than ``"batched_autograd"`` but uses less memory.
- ``"functional_revrev"`` - uses ``torch.autograd.functional`` with "reverse-over-reverse" strategy and a for-loop. This is generally equivalent to ``"autograd"``.
- ``"functional_fwdrev"`` - uses ``torch.autograd.functional`` with vectorized "forward-over-reverse" strategy. Faster than ``"functional_fwdrev"`` but uses more memory (``"batched_autograd"`` seems to be faster)
- ``"func"`` - uses ``torch.func.hessian`` which uses "forward-over-reverse" strategy. This method is the fastest and is recommended, however it is more restrictive and fails with some operators which is why it isn't the default.
- ``"gfd_forward"`` - computes ``ndim`` hessian-vector products via gradient finite difference using a less accurate forward formula which requires one extra gradient evaluation per hessian-vector product.
- ``"gfd_central"`` - computes ``ndim`` hessian-vector products via gradient finite difference using a more accurate central formula which requires two gradient evaluations per hessian-vector product.
- ``"fd"`` - uses function values to estimate gradient and hessian via finite difference. Only computes upper triangle of the hessian, requires ``2n^2 + 1`` function evaluations. This uses less evaluations than chaining ``"gfd_*"`` after ``tz.m.FDM``.
- ``"fd_full"`` - uses function values to estimate gradient and hessian via finite difference. Computes both upper and lower triangles and averages them, requires ``4n^2 - 2n + 1`` function evaluations This uses less evaluations than chaining ``"gfd_*"`` after ``tz.m.FDM``.
- ``"thoad"`` - uses [thoad](https://github.com/mntsx/thoad) library (experimental).

Defaults to ``"batched_autograd"``.
"""

DerivativesMethod = Literal["autograd", "batched_autograd", "thoad"]
"""
Determines how higher order derivatives are computed.
"""

class Objective:
    """
    Holds parameters, gradient, update, objective function (closure) if supplied, loss, and some other info.
    Modules take in a ``Objective`` object, modify and it is passed to the next module.

    Args:
        params (Iterable[torch.Tensor]): iterable of parameters that are being optimized.
        closure (Callable | None, optional): callable that re-evaluates loss. Defaults to None.
        loss (torch.Tensor | None, optional): loss at ``params``. Defaults to None.
        model (torch.nn.Module | None, optional):
            ``torch.nn.Module`` object, needed for a few modules that require access to the model. Defaults to None.
        current_step (int, optional):
            number of times ``Optimizer.step()`` has been called, starting at 0. Defaults to 0.
        parent (Objective | None, optional):
            parent ``Objective`` object. When ``self.get_grad()`` is called, it will also set ``parent.grad``.
            Same with ``self.get_loss()``. This is useful when ``self.params`` are different from ``parent.params``,
            e.g. when projecting. Defaults to None.
        modular (Optimizer | None, optional):
            Top-level ``Optimizer`` optimizer. Defaults to None.
        storage (dict | None, optional):
            additional kwargs passed to ``step`` to control some module-specific behavior. Defaults to None.

    """
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        closure: Callable | None = None,
        loss: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        current_step: int = 0,
        parent: "Objective | None" = None,
        modular: "Optimizer | None" = None,
        storage: dict | None = None,
    ):
        self.params: list[torch.Tensor] = list(params)
        """List of all parameters with ``requires_grad = True``."""

        self.closure = closure
        """A closure that reevaluates the model and returns the loss, None if it wasn't specified"""

        self.model = model
        """``torch.nn.Module`` object of the model, ``None`` if it wasn't specified."""

        self.current_step: int = current_step
        """global current step, starts at 0. This may not correspond to module current step,
        for example a module may step every 10 global steps."""

        self.parent: "Objective | None" = parent
        """parent ``Objective`` object. When ``self.get_grad()`` is called, it will also set ``parent.grad``.
        Same with ``self.get_loss()``. This is useful when ``self.params`` are different from ``parent.params``,
        e.g. when projecting."""

        self.modular: "Optimizer | None" = modular
        """Top-level ``Optimizer`` optimizer, ``None`` if it wasn't specified."""

        self.updates: list[torch.Tensor] | None = None
        """
        current updates list. Update is assumed to be a transformed gradient, therefore it is subtracted.

        If closure is None, this is initially set to cloned gradient. Otherwise this is set to None.

        At the end ``objective.get_update()`` is subtracted from parameters.
        Therefore if ``objective.update`` is ``None``, gradient will be used and calculated if needed.
        """

        self.grads: list[torch.Tensor] | None = None
        """gradient with current parameters. If closure is not ``None``,
        this is set to ``None`` and can be calculated if needed."""

        self.loss: torch.Tensor | Any | None = loss
        """loss with current parameters."""

        self.loss_approx: torch.Tensor | Any | None = None
        """loss at a point near current point. This can be useful as some modules only calculate loss at perturbed points,
        whereas some other modules require loss strictly at current point."""

        self.post_step_hooks: "list[Callable[[Objective, tuple[Module, ...]], None]]" = []
        """list of functions to be called after optimizer step.

        This attribute should always be modified in-place (using ``append`` or ``extend``).

        The signature is:

        ```python
        def hook(objective: Objective, modules: tuple[Module]): ...
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
        """attributes, ``Optimizer.attrs`` is updated with this after each step.
        This attribute should always be modified in-place"""

        if storage is None: storage = {}
        self.storage: dict = storage
        """additional kwargs passed to ``step`` to control some module-specific behavior.
        This attribute should always be modified in-place"""

        self.should_terminate: bool | None = None
        """termination criteria, ``Optimizer.should_terminate`` is set to this after each step if not ``None``"""

        self.temp: Any = cast(Any, None)
        """temporary storage, ``Module.update`` can set this and ``Module.apply`` access via ``objective.poptemp()``.
        This doesn't get cloned."""

    def get_loss(self, backward: bool, retain_graph = None, create_graph: bool = False, at_x0:bool=True) -> torch.Tensor:
        """Returns the loss at current parameters, computing it if it hasn't been computed already
        and assigning ``objective.loss``.Do not call this at perturbed parameters.
        Backward always sets grads to None before recomputing.

        If ``backward==True``, closure is called within ``torch.enable_grad()``
        """

        # at non-x0 point just call closure and return
        if not at_x0:
            if self.closure is None: raise RuntimeError("closure is None")
            return _closure_backward(
                self.closure, self.params, backward=backward, retain_graph=retain_graph, create_graph=create_graph,
            )

        # at x0 set self.loss and self.grads
        if self.loss is None:

            if self.closure is None: raise RuntimeError("closure is None")

            # backward
            if backward:
                self.loss = self.loss_approx = _closure_backward(
                    closure=self.closure, params=self.params, backward=True, retain_graph=retain_graph, create_graph=create_graph
                )

                # next time closure() is called, it will set grad to None.
                # zero_grad(set_to_none=False) shouldn't be used (I should add a warning)
                # because otherwise it will zero self.grads in-place
                self.grads = [p.grad if p.grad  is not None else torch.zeros_like(p) for p in self.params]

            # no backward
            else:
                self.loss = self.loss_approx = _closure_backward(
                    closure=self.closure, params=self.params, backward=False, retain_graph=False, create_graph=False
                )

        # if self.loss was not None, above branch wasn't executed because loss has already been evaluated, but without backward since self.grad is None.
        # and now it is requested to be evaluated with backward.
        if backward and self.grads is None:
            warnings.warn('get_loss was called with backward=False, and then with backward=True so it had to be re-evaluated, so the closure was evaluated twice where it could have been evaluated once.')
            if self.closure is None: raise RuntimeError("closure is None")

            self.loss = self.loss_approx = _closure_backward(
                closure=self.closure, params=self.params, backward=True, retain_graph=retain_graph, create_graph=create_graph
            )
            self.grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]

        # set parent grad
        if self.parent is not None:
            # the way projections/split work, they make a new closure which evaluates original
            # closure and projects the gradient, and set it as their objective.closure.
            # then on `get_loss(backward=True)` it is called, so it also sets original parameters gradient.
            # and we set it to parent objective here.
            if self.parent.loss is None: self.parent.loss = self.loss
            if self.parent.grads is None and backward:
                if all(p.grad is None for p in self.parent.params):
                    warnings.warn("Parent grad is None after backward.")
                self.parent.grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.parent.params]

        return self.loss # type:ignore

    def get_grads(self, retain_graph: bool | None = None, create_graph: bool = False, at_x0: bool = True) -> list[torch.Tensor]:
        """Returns the gradient at initial parameters, computing it if it hasn't been computed already and assigning ``objective.grad`` and potentially ``objective.loss``. Do not call this at perturbed parameters."""
        # at non-x0 point just call closure and return grads
        if not at_x0:
            _, grads = _closure_loss_grad(self.closure, self.params, retain_graph=retain_graph, create_graph=create_graph)
            return grads

        # at x0 get_loss sets self.loss and self.grads
        if self.grads is None:
            if self.closure is None: raise RuntimeError("closure is None")
            self.get_loss(backward=True, retain_graph=retain_graph, create_graph=create_graph) # evaluate and set self.loss and self.grad

        assert self.grads is not None
        return self.grads


    def get_loss_grads(self, retain_graph: bool | None = None, create_graph: bool = False, at_x0: bool = True) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """returns ``(loss, grads)``. Useful when you need both not at x0."""
        # at non-x0 point just call closure and return (loss, grads)
        if not at_x0:
            return _closure_loss_grad(self.closure, self.params, retain_graph=retain_graph, create_graph=create_graph)

        # at x0 get_grads sets self.loss and self.grads, then get_loss returns self.loss.
        grad = self.get_grads(retain_graph=retain_graph, create_graph=create_graph)
        loss = self.get_loss(False)
        return loss, grad

    def get_updates(self) -> list[torch.Tensor]:
        """Returns the update. If update is None, it is initialized by cloning the gradients
        and assigning to ``objective.update``. Computing the gradients may assign ``objective.grad``
        and ``objective.loss`` if they haven't been computed. Do not call this at perturbed parameters."""
        if self.updates is None: self.updates = [g.clone() for g in self.get_grads()]
        return self.updates

    def clone(self, clone_updates: bool, parent: "Objective | None" = None):
        """Creates a shallow copy of this ``Objective``, update can optionally be deep-copied (via ``torch.clone``).

        This copies over all attributes except ``temp``.

        Setting ``parent`` is only if clone's parameters are something different,
        while clone's closure referes to the same objective but with a "view" on parameters.
        """
        copy = Objective(
            params=self.params, closure=self.closure, model=self.model, current_step=self.current_step,
            parent=parent, modular=self.modular, loss=self.loss, storage=self.storage
        )

        if clone_updates and self.updates is not None:
            copy.updates = [u.clone() for u in self.updates]
        else:
            copy.updates = self.updates

        copy.grads = self.grads
        copy.loss_approx = self.loss_approx
        copy.post_step_hooks = self.post_step_hooks
        copy.stop = self.stop
        copy.skip_update = self.skip_update

        copy.attrs = self.attrs
        copy.should_terminate = self.should_terminate

        return copy

    def update_attrs_from_clone_(self, objective: "Objective"):
        """Updates attributes of this ``Objective`` instance from a cloned instance.
        Typically called after a child module has processed a cloned ``Objective``
        object. This propagates any newly computed loss or gradient values
        from the child's context back to the parent ``Objective`` if the parent
        didn't have them computed already.

        This copies over ``loss``, ``loss_approx``, ``grads``, ``should_terminate`` and ``skip_update``.

        Also, as long as ``post_step_hooks`` and ``attrs`` are modified in-place,
        if the child updates them, the update will affect the parent too.
        """
        if self.loss is None: self.loss = objective.loss
        if self.loss_approx is None: self.loss_approx = objective.loss_approx
        if self.grads is None: self.grads = objective.grads

        if objective.should_terminate is not None: self.should_terminate = objective.should_terminate
        if objective.skip_update: self.skip_update = objective.skip_update

    @torch.no_grad
    def zero_grad(self, set_to_none=True):
        """In most cases not call with ``set_to_none=False``, as that will zero ``self.grads`` in-place."""
        if set_to_none:
            for p in self.params: p.grad = None
        else:
            grads = [p.grad for p in self.params if p.grad is not None]
            if len(grads) != 0: torch._foreach_zero_(grads)

    def poptemp(self):
        """to pass information from ``update`` to ``apply``."""
        temp = self.temp
        self.temp = None
        return temp

    @torch.no_grad
    def update_parameters(self):
        """subtracts ``self.get_updates()`` from parameters, unless ``self.skip_update = True``, then does nothing."""
        if self.skip_update: return
        torch._foreach_sub_(self.params, self.get_updates())

    def apply_post_step_hooks(self, modules: "Sequence[Module]"):
        """Runs hooks that a few modules use. This should be called **after** updating parameters."""
        modules = tuple(modules)
        for hook in self.post_step_hooks:
            hook(self, modules)


    # ------------------------------ HELPER METHODS ------------------------------ #
    @torch.no_grad
    def hessian_vector_product(
        self,
        z: Sequence[torch.Tensor],
        rgrad: Sequence[torch.Tensor] | None,
        at_x0: bool,
        hvp_method: HVPMethod,
        h: float,
        retain_graph: bool = False,
    ) -> tuple[list[torch.Tensor], Sequence[torch.Tensor] | None]:
        """
        Returns ``(Hz, rgrad)``, where ``rgrad`` is gradient at current parameters but it may be None.

        Gradient is set to ``objective`` automatically if ``at_x0`` and can be accessed with ``objective.get_grad()``.

        Single hessian vector product example:

        ```python
        Hz, _ = self.hessian_vector_product(z, rgrad=None, at_x0=True, ..., retain_graph=False)
        ```

        Multiple hessian-vector products example:

        ```python
        rgrad = None
        for z in vecs:
            retain_graph = i < len(vecs) - 1
            Hz, rgrad = self.hessian_vector_product(z, rgrad=rgrad, ..., retain_graph=retain_graph)

        ```

        Args:
            z (Sequence[torch.Tensor]): vector in hessian-vector product
            rgrad (Sequence[torch.Tensor] | None): pass None initially, then pass what this returns.
            at_x0 (bool): whether this is being called at original or perturbed parameters.
            hvp_method (str): hvp method.
            h (float): finite difference step size
            retain_grad (bool): retain grad
        """
        if hvp_method in ('batched_autograd', "autograd"):
            with torch.enable_grad():
                if rgrad is None: rgrad = self.get_grads(create_graph=True, at_x0=at_x0)
                Hz = torch.autograd.grad(rgrad, self.params, z, retain_graph=retain_graph)

        # loss returned by fd hvp is not guaranteed to be at x0 so we don't use/return it
        elif hvp_method == 'fd_forward':
            if rgrad is None: rgrad = self.get_grads(at_x0=at_x0)
            _, Hz = hvp_fd_forward(self.closure, self.params, z, h=h, g_0=rgrad)

        elif hvp_method == 'fd_central':
            _, Hz = hvp_fd_central(self.closure, self.params, z, h=h)

        else:
            raise ValueError(hvp_method)

        return list(Hz), rgrad

    @torch.no_grad
    def hessian_matrix_product(
        self,
        Z: torch.Tensor,
        rgrad: Sequence[torch.Tensor] | None,
        at_x0: bool,
        hvp_method: HVPMethod,
        h: float,
        retain_graph: bool = False,
    ) -> tuple[torch.Tensor, Sequence[torch.Tensor] | None]:
        """Z is ``(n_dim, n_hvps)``, computes ``H @ Z`` of shape ``(n_dim, n_hvps)``.

        Returns ``(HZ, rgrad)`` where ``rgrad`` is gradient at current parameters but it may be None.

        Gradient is set to ``objective`` automatically if ``at_x0`` and can be accessed with ``objective.get_grad()``.

        Unlike ``hessian_vector_product`` this returns a single matrix, not a per-parameter list.

        Args:
            Z (torch.Tensor): matrix in hessian-matrix product
            rgrad (Sequence[torch.Tensor] | None): pass None initially, then pass what this returns.
            at_x0 (bool): whether this is being called at original or perturbed parameters.
            hvp_method (str): hvp method.
            h (float): finite difference step size
            retain_grad (bool): retain grad

        """
        # compute
        if hvp_method == "batched_autograd":
            with torch.enable_grad():
                if rgrad is None: rgrad = self.get_grads(create_graph=True, at_x0=at_x0)
                flat_inputs = torch.cat([g.ravel() for g in rgrad])
                HZ_list = torch.autograd.grad(
                    flat_inputs,
                    self.params,
                    grad_outputs=Z.T,
                    is_grads_batched=True,
                    retain_graph=retain_graph,
                )

            HZ = flatten_jacobian(HZ_list).T

        elif hvp_method == 'autograd':
            with torch.enable_grad():
                if rgrad is None: rgrad = self.get_grads(create_graph=True, at_x0=at_x0)
                flat_inputs = torch.cat([g.ravel() for g in rgrad])
                HZ_tensors = [
                    torch.autograd.grad(
                        flat_inputs,
                        self.params,
                        grad_outputs=col,
                        retain_graph=retain_graph or (i < Z.size(1) - 1),
                    )
                    for i, col in enumerate(Z.unbind(1))
                ]

            HZ_list = [torch.cat([t.ravel() for t in tensors]) for tensors in HZ_tensors]
            HZ = torch.stack(HZ_list, 1)

        elif hvp_method == 'fd_forward':
            if rgrad is None: rgrad = self.get_grads(at_x0=at_x0)
            HZ_tensors = [
                hvp_fd_forward(
                    self.closure,
                    self.params,
                    vec_to_tensors(col, self.params),
                    h=h,
                    g_0=rgrad,
                )[1]
                for col in Z.unbind(1)
            ]
            HZ_list = [torch.cat([t.ravel() for t in tensors]) for tensors in HZ_tensors]
            HZ = flatten_jacobian(HZ_list)

        elif hvp_method == 'fd_central':
            HZ_tensors = [
                hvp_fd_central(
                    self.closure, self.params, vec_to_tensors(col, self.params), h=h
                )[1]
                for col in Z.unbind(1)
            ]
            HZ_list = [torch.cat([t.ravel() for t in tensors]) for tensors in HZ_tensors]
            HZ = flatten_jacobian(HZ_list)

        else:
            raise ValueError(hvp_method)

        return HZ, rgrad

    @torch.no_grad
    def hutchinson_hessian(
        self,
        rgrad: Sequence[torch.Tensor] | None,
        at_x0: bool,
        n_samples: int | None,
        distribution: Distributions | Sequence[Sequence[torch.Tensor]],
        hvp_method: HVPMethod,
        h: float,
        generator,
        variance: int | None = 1,
        zHz: bool = True,
        retain_graph: bool = False,
    ) -> tuple[list[torch.Tensor], Sequence[torch.Tensor] | None]:
        """
        Returns ``(D, rgrad)``, where ``rgrad`` is gradient at current parameters but it may be None.

        Gradient is set to ``objective`` automatically if ``at_x0`` and can be accessed with ``objective.get_grad()``.

        Args:
            rgrad (Sequence[torch.Tensor] | None): pass None initially, then pass what this returns.
            at_x0 (bool): whether this is being called at original or perturbed parameters.
            n_samples (int | None): number of random vectors.
            distribution (Distributions | Sequence[Sequence[torch.Tensor]]):
                distribution, this can also be a sequence of tensor sequences.
            hvp_method (str): how to compute hessian-vector products.
            h (float): finite difference step size.
            generator (Any): generator
            variance (int | None, optional): variance of random vectors. Defaults to 1.
            zHz (bool, optional): whether to compute z ⊙ Hz. If False, computes Hz. Defaults to True.
            retain_graph (bool, optional): whether to retain graph. Defaults to False.
        """

        params = TensorList(self.params)
        samples = None

        # check when distribution is sequence of tensors
        if not isinstance(distribution, str):
            if n_samples is not None and n_samples != len(distribution):
                raise RuntimeError("when passing sequence of z to `hutchinson_hessian`, set `n_samples` to None")

            n_samples = len(distribution)
            samples = distribution

        # use non-batched with single sample
        if n_samples == 1 and hvp_method == 'batched_autograd':
            hvp_method = 'autograd'

        # -------------------------- non-batched hutchinson -------------------------- #
        if hvp_method in ('autograd', 'fd_forward', 'fd_central'):

            D = None
            assert n_samples is not None

            for i in range(n_samples):

                # sample
                if samples is not None: z = samples[i]
                else: z = params.sample_like(cast(Distributions, distribution), variance, generator=generator)

                # compute
                Hz, rgrad = self.hessian_vector_product(
                    z=z,
                    rgrad=rgrad,
                    at_x0=at_x0,
                    hvp_method=hvp_method,
                    h=h,
                    retain_graph=(i < n_samples - 1) or retain_graph,
                )

                # add
                if zHz: torch._foreach_mul_(Hz, tuple(z))

                if D is None: D = Hz
                else: torch._foreach_add_(D, Hz)


            assert D is not None
            if n_samples > 1: torch._foreach_div_(D, n_samples)
            return D, rgrad

        # ---------------------------- batched hutchinson ---------------------------- #
        if hvp_method != 'batched_autograd':
            raise RuntimeError(f"Unknown hvp_method: `{hvp_method}`")

        # generate and vectorize samples
        if samples is None:
            samples = [params.sample_like(cast(Distributions, distribution), variance, generator=generator).to_vec()]

        else:
            samples = [torch.cat([t.ravel() for t in s]) for s in samples]

        # compute Hz
        Z = torch.stack(samples, -1)
        HZ, rgrad = self.hessian_matrix_product(
            Z,
            rgrad=rgrad,
            at_x0=at_x0,
            hvp_method='batched_autograd',
            h=h, # not used
            retain_graph=retain_graph,
        )

        if zHz: HZ *= Z
        D_vec = HZ.mean(-1)
        return vec_to_tensors(D_vec, params), rgrad

    @torch.no_grad
    def hessian(
        self,
        hessian_method: HessianMethod,
        h: float,
        at_x0: bool,
    ) -> tuple[torch.Tensor | None, Sequence[torch.Tensor] | None, torch.Tensor]:
        """returns ``(f, g_list, H)``. Also sets ``objective.loss`` and ``objective.grad`` if ``at_x0``.

        ``f`` and ``g_list`` may be None if they aren't computed with ``hessian_method``.

        Args:
            hessian_method: how to compute hessian
            h (float): finite difference step size
            vectorize (bool): whether to vectorize hessian computation
            at_x0 (bool): whether its at x0.
        """
        closure = self.closure
        if closure is None:
            raise RuntimeError("Computing hessian requires a closure to be provided to the `step` method.")

        params = self.params
        numel = sum(p.numel() for p in params)

        f = None
        g_list = None

        # autograd hessian
        if hessian_method in ("batched_autograd", "autograd"):
            with torch.enable_grad():
                f = self.get_loss(False, at_x0=at_x0)

                batched = hessian_method == "batched_autograd"
                g_list, H_list = jacobian_and_hessian_wrt([f.ravel()], params, batched=batched)
                g_list = [t[0] for t in g_list] # remove leading dim from loss

            H = flatten_jacobian(H_list)

        # functional autograd hessian
        elif hessian_method in ('func', 'functional_revrev', 'functional_fwdrev'):
            if hessian_method == 'functional_fwdrev':
                method = "autograd.functional"
                outer_jacobian_strategy = "forward-mode"
                vectorize=True
            elif hessian_method == 'functional_revrev':
                method = "autograd.functional"
                outer_jacobian_strategy = "reverse-mode"
                vectorize=False
            else:
                method = 'func'
                outer_jacobian_strategy = "forward-mode" # unused
                vectorize=True # unused

            with torch.enable_grad():
                H = hessian_mat(partial(closure, backward=False), params,
                                method=method, vectorize=vectorize,
                                outer_jacobian_strategy=outer_jacobian_strategy)

        # thoad
        elif hessian_method == "thoad":
            with torch.enable_grad():
                f = self.get_loss(False, at_x0=at_x0)
                ctrl = lazy_thoad.backward(f, 2, crossings=True)

            g_list = [p.hgrad[0].squeeze(0) for p in params] # pyright:ignore[reportAttributeAccessIssue]
            H = thoad_single_tensor(ctrl, params, 2)


        # gradient finite difference
        elif hessian_method in ('gfd_forward', 'gfd_central'):

            if hessian_method == 'gfd_central': hvp_method = 'fd_central'
            else: hvp_method = 'fd_forward'

            I = torch.eye(numel, device=params[0].device, dtype=params[0].dtype)
            H, g_list = self.hessian_matrix_product(I, rgrad=None, at_x0=at_x0, hvp_method=hvp_method, h=h)

        # function value finite difference
        elif hessian_method in ('fd', "fd_full"):
            full = hessian_method == "fd_full"
            f, g_list, H = hessian_fd(partial(closure, False), params=params, eps=h, full=full)

        else:
            raise ValueError(hessian_method)

        # set objective attributes if at x0
        if at_x0:
            if f is not None and self.loss is None:
                self.loss = self.loss_approx = f

            if g_list is not None and self.grads is None:
                self.grads = list(g_list)

        return f, g_list, H.detach()

    @torch.no_grad
    def derivatives(self, order: int, at_x0: bool, method:DerivativesMethod="batched_autograd"):
        """
        returns a tuple of tensors of function value and derivatives up to ``order``

        ``order = 0`` returns ``(f,)``;

        ``order = 1`` returns ``(f, g)``;

        ``order = 2`` returns ``(f, g, H)``;

        ``order = 3`` returns ``(f, g, H, T3)``;

        etc.
        """
        closure = self.closure
        if closure is None:
            raise RuntimeError("Computing hessian requires a closure to be provided to the `step` method.")

        # just loss
        if order == 0:
            f = self.get_loss(False, at_x0=at_x0)
            return (f, )

        # loss and grad
        if order == 1:
            f, g_list = self.get_loss_grads(at_x0=at_x0)
            g = torch.cat([t.ravel() for t in g_list])

            return f, g

        if method in ("autograd", "batched_autograd"):
            batched = method == "batched_autograd"

            # recursively compute derivatives up to order
            with torch.enable_grad():
                f, g_list = self.get_loss_grads(at_x0=at_x0, create_graph=True)
                g = torch.cat([t.ravel() for t in g_list])

                n = g.numel()
                ret = [f, g]
                T = g # current derivatives tensor

                # get all derivative up to order
                for o in range(2, order + 1):
                    is_last = o == order
                    T_list = jacobian_wrt([T], self.params, create_graph=not is_last, batched=batched)
                    with torch.no_grad() if is_last else nullcontext():

                        # the shape is (ndim, ) * order
                        T = flatten_jacobian(T_list).view(n, n, *T.shape[1:])
                        ret.append(T)

            return tuple(ret)

        if method == "thoad":
            with torch.enable_grad():
                f = self.get_loss(False, at_x0=at_x0)
                ctrl = lazy_thoad.backward(f, order, crossings=True)

                return tuple([f, *thoad_derivatives(ctrl, self.params, order=order)])

        raise ValueError(method)

    @torch.no_grad
    def derivatives_at(
        self,
        x: torch.Tensor | Sequence[torch.Tensor],
        order: int,
        method:DerivativesMethod="batched_autograd"
    ):
        """
        returns a tuple of tensors of function value and derivatives up to ``order`` at ``x``,
        then sets original parameters.

        ``x`` can be a vector or a list of tensors.

        ``order = 0`` returns ``(f,)``;

        ``order = 1`` returns ``(f, g)``;

        ``order = 2`` returns ``(f, g, H)``;

        ``order = 3`` returns ``(f, g, H, T3)``;

        etc.
        """
        if isinstance(x, torch.Tensor): x = vec_to_tensors(x, self.params)

        x0 = [p.clone() for p in self.params]

        # set params to x
        for p, x_i in zip(self.params, x):
            set_storage_(p, x_i)

        ret = self.derivatives(order=order, at_x0=False, method=method)

        # set params to x0
        for p, x0_i in zip(self.params, x0):
            set_storage_(p, x0_i)

        return ret


    def list_Hvp_function(self, hvp_method: HVPMethod, h: float, at_x0:bool):
        """returns ``(grad, H_mv)`` where ``H_mv`` is a callable that accepts and returns lists of tensors.

        ``grad`` may be None, and this sets ``objective.grad`` if ``at_x0`` so at x0 just use ``objective.get_grad()``.
        """
        params = TensorList(self.params)
        closure = self.closure

        if hvp_method in ('batched_autograd', 'autograd'):
            grad = self.get_grads(create_graph=True, at_x0=at_x0)

            def H_mv(x: torch.Tensor | Sequence[torch.Tensor]):
                if isinstance(x, torch.Tensor): x = params.from_vec(x)
                with torch.enable_grad():
                    return TensorList(torch.autograd.grad(grad, params, x, retain_graph=True))

        else:

            if hvp_method == 'fd_forward':
                grad = self.get_grads(at_x0=at_x0)
                def H_mv(x: torch.Tensor | Sequence[torch.Tensor]):
                    if isinstance(x, torch.Tensor): x = params.from_vec(x)
                    _, Hx = hvp_fd_forward(closure, params, x, h=h, g_0=grad)
                    return TensorList(Hx)

            elif hvp_method == 'fd_central':
                grad = None
                def H_mv(x: torch.Tensor | Sequence[torch.Tensor]):
                    if isinstance(x, torch.Tensor): x = params.from_vec(x)
                    _, Hx = hvp_fd_central(closure, params, x, h=h)
                    return TensorList(Hx)

            else:
                raise ValueError(hvp_method)


        return grad, H_mv

    def tensor_Hvp_function(self, hvp_method: HVPMethod, h: float, at_x0:bool):
        """returns ``(grad, H_mv, H_mm)``, where ``H_mv`` and ``H_mm`` accept and return single tensors.

        ``grad`` may be None, and this sets ``objective.grad`` if ``at_x0`` so at x0 just use ``objective.get_grad()``.
        """
        if hvp_method in ('fd_forward', "fd_central", "autograd"):
            grad, list_H_mv = self.list_Hvp_function(hvp_method=hvp_method, h=h, at_x0=at_x0)

            def H_mv_loop(x: torch.Tensor):
                Hx_list = list_H_mv(x)
                return torch.cat([t.ravel() for t in Hx_list])

            def H_mm_loop(X: torch.Tensor):
                return torch.stack([H_mv_loop(col) for col in X.unbind(-1)], -1)

            return grad, H_mv_loop, H_mm_loop

        # for batched we need grad
        if hvp_method != 'batched_autograd':
            raise RuntimeError(f"Unknown hvp_method `{hvp_method}`")

        params = TensorList(self.params)
        grad = self.get_grads(create_graph=True, at_x0=at_x0)

        def H_mv_batched(x: torch.Tensor):
            with torch.enable_grad():
                Hx_list = torch.autograd.grad(grad, params, params.from_vec(x), retain_graph=True)

            return torch.cat([t.ravel() for t in Hx_list])

        def H_mm_batched(X: torch.Tensor):
            with torch.enable_grad():
                flat_inputs = torch.cat([g.ravel() for g in grad])
                HX_list = torch.autograd.grad(
                    flat_inputs,
                    self.params,
                    grad_outputs=X.T,
                    is_grads_batched=True,
                    retain_graph=True,
                )
            return flatten_jacobian(HX_list).T

        return grad, H_mv_batched, H_mm_batched


# endregion
