from collections.abc import Iterable, Sequence

import torch
import torch.autograd.forward_ad as fwAD
from typing import Literal

from .torch_tools import swap_tensors_no_use_count_check, vec_to_tensors

def _jacobian(output: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False):
    flat_input = torch.cat([i.reshape(-1) for i in output])
    grad_ouputs = torch.eye(len(flat_input), device=output[0].device, dtype=output[0].dtype)
    jac = []
    for i in range(flat_input.numel()):
        jac.append(torch.autograd.grad(
            flat_input,
            wrt,
            grad_ouputs[i],
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
            is_grads_batched=False,
        ))
    return [torch.stack(z) for z in zip(*jac)]


def _jacobian_batched(output: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False):
    flat_input = torch.cat([i.reshape(-1) for i in output])
    return torch.autograd.grad(
        flat_input,
        wrt,
        torch.eye(len(flat_input), device=output[0].device, dtype=output[0].dtype),
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=True,
        is_grads_batched=True,
    )

def jacobian_wrt(output: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False, batched=True) -> Sequence[torch.Tensor]:
    """Calculate jacobian of a sequence of tensors w.r.t another sequence of tensors.
    Returns a sequence of tensors with the length as `wrt`.
    Each tensor will have the shape `(*input.shape, *wrt[i].shape)`.

    Args:
        input (Sequence[torch.Tensor]): input sequence of tensors.
        wrt (Sequence[torch.Tensor]): sequence of tensors to differentiate w.r.t.
        create_graph (bool, optional):
            pytorch option, if True, graph of the derivative will be constructed,
            allowing to compute higher order derivative products. Default: False.
        batched (bool, optional): use faster but experimental pytorch batched jacobian
            This only has effect when `input` has more than 1 element. Defaults to True.

    Returns:
        sequence of tensors with the length as `wrt`.
    """
    if batched: return _jacobian_batched(output, wrt, create_graph)
    return _jacobian(output, wrt, create_graph)

def jacobian_and_hessian_wrt(output: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False, batched=True):
    """Calculate jacobian and hessian of a sequence of tensors w.r.t another sequence of tensors.
    Calculating hessian requires calculating the jacobian. So this function is more efficient than
    calling `jacobian` and `hessian` separately, which would calculate jacobian twice.

    Args:
        input (Sequence[torch.Tensor]): input sequence of tensors.
        wrt (Sequence[torch.Tensor]): sequence of tensors to differentiate w.r.t.
        create_graph (bool, optional):
            pytorch option, if True, graph of the derivative will be constructed,
            allowing to compute higher order derivative products. Default: False.
        batched (bool, optional): use faster but experimental pytorch batched grad. Defaults to True.

    Returns:
        tuple with jacobians sequence and hessians sequence.
    """
    jac = jacobian_wrt(output, wrt, create_graph=True, batched = batched)
    return jac, jacobian_wrt(jac, wrt, batched = batched, create_graph=create_graph)


def hessian_list_to_mat(hessians: Sequence[torch.Tensor]):
    """takes output of `hessian` and returns the 2D hessian matrix.
    Note - I only tested this for cases where input is a scalar."""
    return torch.cat([h.reshape(h.size(0), h[1].numel()) for h in hessians], 1)

def jacobian_and_hessian_mat_wrt(output: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False, batched=True):
    """Calculate jacobian and hessian of a sequence of tensors w.r.t another sequence of tensors.
    Calculating hessian requires calculating the jacobian. So this function is more efficient than
    calling `jacobian` and `hessian` separately, which would calculate jacobian twice.

    Args:
        input (Sequence[torch.Tensor]): input sequence of tensors.
        wrt (Sequence[torch.Tensor]): sequence of tensors to differentiate w.r.t.
        create_graph (bool, optional):
            pytorch option, if True, graph of the derivative will be constructed,
            allowing to compute higher order derivative products. Default: False.
        batched (bool, optional): use faster but experimental pytorch batched grad. Defaults to True.

    Returns:
        tuple with jacobians sequence and hessians sequence.
    """
    jac = jacobian_wrt(output, wrt, create_graph=True, batched = batched)
    H_list = jacobian_wrt(jac, wrt, batched = batched, create_graph=create_graph)
    return torch.cat([j.view(-1) for j in jac]), hessian_list_to_mat(H_list)

def hessian(
    fn,
    params: Iterable[torch.Tensor],
    create_graph=False,
    method="func",
    vectorize=False,
    outer_jacobian_strategy="reverse-mode",
):
    """
    returns list of lists of lists of values of hessian matrix of each param wrt each param.
    To just get a single matrix use the :code:`hessian_mat` function.

    `vectorize` and `outer_jacobian_strategy` are only for `method = "torch.autograd"`, refer to its documentation.

    Example:
    .. code:: py

        model = nn.Linear(4, 2) # (2, 4) weight and (2, ) bias
        X = torch.randn(10, 4)
        y = torch.randn(10, 2)

        def fn():
            y_hat = model(X)
            loss = F.mse_loss(y_hat, y)
            return loss

        hessian_mat(fn, model.parameters()) # list of two lists of two lists of 3D and 4D tensors


    """
    params = list(params)

    def func(x: list[torch.Tensor]):
        for p, x_i in zip(params, x): swap_tensors_no_use_count_check(p, x_i)
        loss = fn()
        for p, x_i in zip(params, x): swap_tensors_no_use_count_check(p, x_i)
        return loss

    if method == 'func':
        return torch.func.hessian(func)([p.detach().requires_grad_(create_graph) for p in params])

    if method == 'autograd.functional':
        return torch.autograd.functional.hessian(
            func,
            [p.detach() for p in params],
            create_graph=create_graph,
            vectorize=vectorize,
            outer_jacobian_strategy=outer_jacobian_strategy,
        )
    raise ValueError(method)

def hessian_mat(
    fn,
    params: Iterable[torch.Tensor],
    create_graph=False,
    method="func",
    vectorize=False,
    outer_jacobian_strategy="reverse-mode",
) -> torch.Tensor:
    """
    returns hessian matrix for parameters (as if they were flattened and concatenated into a vector).

    `vectorize` and `outer_jacobian_strategy` are only for `method = "torch.autograd"`, refer to its documentation.

    Example:
    .. code:: py

        model = nn.Linear(4, 2) # 10 parameters in total
        X = torch.randn(10, 4)
        y = torch.randn(10, 2)

        def fn():
            y_hat = model(X)
            loss = F.mse_loss(y_hat, y)
            return loss

        hessian_mat(fn, model.parameters()) # 10x10 tensor


    """
    params = list(params)

    def func(x: torch.Tensor):
        x_params = vec_to_tensors(x, params)
        for p, x_i in zip(params, x_params): swap_tensors_no_use_count_check(p, x_i)
        loss = fn()
        for p, x_i in zip(params, x_params): swap_tensors_no_use_count_check(p, x_i)
        return loss

    if method == 'func':
        return torch.func.hessian(func)(torch.cat([p.view(-1) for p in params]).detach().requires_grad_(create_graph)) # pyright:ignore[reportReturnType]

    if method == 'autograd.functional':
        return torch.autograd.functional.hessian(
            func,
            torch.cat([p.view(-1) for p in params]).detach(),
            create_graph=create_graph,
            vectorize=vectorize,
            outer_jacobian_strategy=outer_jacobian_strategy,
        ) # pyright:ignore[reportReturnType]
    raise ValueError(method)

def jvp(fn, params: Iterable[torch.Tensor], tangent: Iterable[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Jacobian vector product.

    Example:
    .. code:: py

        model = nn.Linear(4, 2)
        X = torch.randn(10, 4)
        y = torch.randn(10, 2)

        tangent = [torch.randn_like(p) for p in model.parameters()]

        def fn():
            y_hat = model(X)
            loss = F.mse_loss(y_hat, y)
            return loss

        jvp(fn, model.parameters(), tangent) # scalar

    """
    params = list(params)
    tangent = list(tangent)
    detached_params = [p.detach() for p in params]

    duals = []
    with fwAD.dual_level():
        for p, d, t in zip(params, detached_params, tangent):
            dual = fwAD.make_dual(d, t).requires_grad_(p.requires_grad)
            duals.append(dual)
            swap_tensors_no_use_count_check(p, dual)

        loss = fn()
        res = fwAD.unpack_dual(loss).tangent

    for p, d in zip(params, duals):
        swap_tensors_no_use_count_check(p, d)
    return loss, res



@torch.no_grad
def jvp_fd_central(
    fn,
    params: Iterable[torch.Tensor],
    tangent: Iterable[torch.Tensor],
    h=1e-3,
    normalize=False,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Jacobian vector product using central finite difference formula.

    Example:
    .. code:: py

        model = nn.Linear(4, 2)
        X = torch.randn(10, 4)
        y = torch.randn(10, 2)

        tangent = [torch.randn_like(p) for p in model.parameters()]

        def fn():
            y_hat = model(X)
            loss = F.mse_loss(y_hat, y)
            return loss

        jvp_fd_central(fn, model.parameters(), tangent) # scalar

    """
    params = list(params)
    tangent = list(tangent)

    tangent_norm = None
    if normalize:
        tangent_norm = torch.linalg.vector_norm(torch.cat([t.view(-1) for t in tangent])) # pylint:disable=not-callable
        if tangent_norm == 0: return None, torch.tensor(0., device=tangent[0].device, dtype=tangent[0].dtype)
        tangent = torch._foreach_div(tangent, tangent_norm)

    tangent_h= torch._foreach_mul(tangent, h)

    torch._foreach_add_(params, tangent_h)
    v_plus = fn()
    torch._foreach_sub_(params, tangent_h)
    torch._foreach_sub_(params, tangent_h)
    v_minus = fn()
    torch._foreach_add_(params, tangent_h)

    res = (v_plus - v_minus) / (2 * h)
    if normalize: res = res * tangent_norm
    return v_plus, res

@torch.no_grad
def jvp_fd_forward(
    fn,
    params: Iterable[torch.Tensor],
    tangent: Iterable[torch.Tensor],
    h=1e-3,
    v_0=None,
    normalize=False,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Jacobian vector product using forward finite difference formula.
    Loss at initial point can be specified in the `v_0` argument.

    Example:
    .. code:: py

        model = nn.Linear(4, 2)
        X = torch.randn(10, 4)
        y = torch.randn(10, 2)

        tangent1 = [torch.randn_like(p) for p in model.parameters()]
        tangent2 = [torch.randn_like(p) for p in model.parameters()]

        def fn():
            y_hat = model(X)
            loss = F.mse_loss(y_hat, y)
            return loss

        v_0 = fn() # pre-calculate loss at initial point

        jvp1 = jvp_fd_forward(fn, model.parameters(), tangent1, v_0=v_0) # scalar
        jvp2 = jvp_fd_forward(fn, model.parameters(), tangent2, v_0=v_0) # scalar

    """
    params = list(params)
    tangent = list(tangent)

    tangent_norm = None
    if normalize:
        tangent_norm = torch.linalg.vector_norm(torch.cat([t.view(-1) for t in tangent])) # pylint:disable=not-callable
        if tangent_norm == 0: return None, torch.tensor(0., device=tangent[0].device, dtype=tangent[0].dtype)
        tangent = torch._foreach_div(tangent, tangent_norm)

    tangent_h= torch._foreach_mul(tangent, h)

    if v_0 is None: v_0 = fn()

    torch._foreach_add_(params, tangent_h)
    v_plus = fn()
    torch._foreach_sub_(params, tangent_h)

    res = (v_plus - v_0) / h
    if normalize: res = res * tangent_norm
    return v_0, res

def hvp(
    params: Iterable[torch.Tensor],
    grads: Iterable[torch.Tensor],
    vec: Iterable[torch.Tensor],
    retain_graph=None,
    create_graph=False,
    allow_unused=None,
):
    """Hessian-vector product

    Example:
    .. code:: py

        model = nn.Linear(4, 2)
        X = torch.randn(10, 4)
        y = torch.randn(10, 2)

        y_hat = model(X)
        loss = F.mse_loss(y_hat, y)
        loss.backward(create_graph=True)

        grads = [p.grad for p in model.parameters()]
        vec = [torch.randn_like(p) for p in model.parameters()]

        # list of tensors, same layout as model.parameters()
        hvp(model.parameters(), grads, vec=vec)
    """
    params = list(params)
    g = list(grads)
    vec = list(vec)

    with torch.enable_grad():
        return torch.autograd.grad(g, params, vec, create_graph=create_graph, retain_graph=retain_graph, allow_unused=allow_unused)


@torch.no_grad
def hvp_fd_central(
    closure,
    params: Iterable[torch.Tensor],
    vec: Iterable[torch.Tensor],
    h=1e-3,
    normalize=False,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    """Hessian-vector product using central finite difference formula.

    Please note that this will clear :code:`grad` attributes in params.

    Example:
    .. code:: py

        model = nn.Linear(4, 2)
        X = torch.randn(10, 4)
        y = torch.randn(10, 2)

        def closure():
            y_hat = model(X)
            loss = F.mse_loss(y_hat, y)
            model.zero_grad()
            loss.backward()
            return loss

        vec = [torch.randn_like(p) for p in model.parameters()]

        # list of tensors, same layout as model.parameters()
        hvp_fd_central(closure, model.parameters(), vec=vec)
    """
    params = list(params)
    vec = list(vec)

    vec_norm = None
    if normalize:
        vec_norm = torch.linalg.vector_norm(torch.cat([t.view(-1) for t in vec])) # pylint:disable=not-callable
        if vec_norm == 0: return None, [torch.zeros_like(p) for p in params]
        vec = torch._foreach_div(vec, vec_norm)

    vec_h = torch._foreach_mul(vec, h)
    torch._foreach_add_(params, vec_h)
    with torch.enable_grad(): loss = closure()
    g_plus = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    torch._foreach_sub_(params, vec_h)
    torch._foreach_sub_(params, vec_h)
    with torch.enable_grad(): loss = closure()
    g_minus = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    torch._foreach_add_(params, vec_h)
    for p in params: p.grad = None

    hvp_ = g_plus
    torch._foreach_sub_(hvp_, g_minus)
    torch._foreach_div_(hvp_, 2*h)

    if normalize: torch._foreach_mul_(hvp_, vec_norm)
    return loss, hvp_

@torch.no_grad
def hvp_fd_forward(
    closure,
    params: Iterable[torch.Tensor],
    vec: Iterable[torch.Tensor],
    h=1e-3,
    g_0=None,
    normalize=False,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    """Hessian-vector product using forward finite difference formula.

    Gradient at initial point can be specified in the `g_0` argument.

    Please note that this will clear :code:`grad` attributes in params.

    Example:
    .. code:: py

        model = nn.Linear(4, 2)
        X = torch.randn(10, 4)
        y = torch.randn(10, 2)

        def closure():
            y_hat = model(X)
            loss = F.mse_loss(y_hat, y)
            model.zero_grad()
            loss.backward()
            return loss

        vec = [torch.randn_like(p) for p in model.parameters()]

        # pre-compute gradient at initial point
        closure()
        g_0 = [p.grad for p in model.parameters()]

        # list of tensors, same layout as model.parameters()
        hvp_fd_forward(closure, model.parameters(), vec=vec, g_0=g_0)
    """

    params = list(params)
    vec = list(vec)
    loss = None

    vec_norm = None
    if normalize:
        vec_norm = torch.linalg.vector_norm(torch.cat([t.view(-1) for t in vec])) # pylint:disable=not-callable
        if vec_norm == 0: return None, [torch.zeros_like(p) for p in params]
        vec = torch._foreach_div(vec, vec_norm)

    vec_h = torch._foreach_mul(vec, h)

    if g_0 is None:
        with torch.enable_grad(): loss = closure()
        g_0 = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    else:
        g_0 = list(g_0)

    torch._foreach_add_(params, vec_h)
    with torch.enable_grad():
        l = closure()
        if loss is None: loss = l
    g_plus = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    torch._foreach_sub_(params, vec_h)
    for p in params: p.grad = None

    hvp_ = g_plus
    torch._foreach_sub_(hvp_, g_0)
    torch._foreach_div_(hvp_, h)

    if normalize: torch._foreach_mul_(hvp_, vec_norm)
    return loss, hvp_
