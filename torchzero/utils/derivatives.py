from collections.abc import Iterable

import torch
import torch.autograd.forward_ad as fwAD

from .torch_tools import swap_tensors_no_use_count_check, vec_to_tensors



def hessian(
    fn,
    params: Iterable[torch.Tensor],
    create_graph=False,
    method="torch.func",
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

    if method == 'torch.func':
        return torch.func.hessian(func)([p.detach().requires_grad_(create_graph) for p in params])

    if method == 'torch.autograd':
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
    method="torch.func",
    vectorize=False,
    outer_jacobian_strategy="reverse-mode",
):
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

    if method == 'torch.func':
        return torch.func.hessian(func)(torch.cat([p.view(-1) for p in params]).detach().requires_grad_(create_graph))

    if method == 'torch.autograd':
        return torch.autograd.functional.hessian(
            func,
            torch.cat([p.view(-1) for p in params]).detach(),
            create_graph=create_graph,
            vectorize=vectorize,
            outer_jacobian_strategy=outer_jacobian_strategy,
        )
    raise ValueError(method)

def jvp(fn, params: Iterable[torch.Tensor], tangent: Iterable[torch.Tensor]):
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
            dual = fwAD.make_dual(d, t)
            duals.append(dual)
            swap_tensors_no_use_count_check(p, dual)

        loss = fn()
        tangent = fwAD.unpack_dual(loss).tangent

    for p, d in zip(params, duals):
        swap_tensors_no_use_count_check(p, d)

    return tangent



@torch.no_grad
def jvp_fd_central(fn, params: Iterable[torch.Tensor], tangent: Iterable[torch.Tensor], h=1e-3):
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
    tangent_h= torch._foreach_mul(tangent, h)

    torch._foreach_add_(params, tangent_h)
    v_plus = fn()
    torch._foreach_sub_(params, tangent_h)
    torch._foreach_sub_(params, tangent_h)
    v_minus = fn()
    torch._foreach_add_(params, tangent_h)

    return (v_plus - v_minus) / (2 * h)

@torch.no_grad
def jvp_fd_forward(fn, params: Iterable[torch.Tensor], tangent: Iterable[torch.Tensor], h=1e-3, v_0 = None):
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
    tangent_h= torch._foreach_mul(tangent, h)

    if v_0 is None: v_0 = fn()

    torch._foreach_add_(params, tangent_h)
    v_plus = fn()
    torch._foreach_sub_(params, tangent_h)

    return (v_plus - v_0) / h

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

        grads = [p.grad for p in model.parameters()]
        vec = [torch.randn_like(p) for p in model.parameters()]

        # list of tensors, same layout as model.parameters()
        hvp(model.parameters(), grads, vec=vec)
    """
    params = list(params)
    g = list(grads)
    vec = list(vec)

    return torch.autograd.grad(g, params, vec, create_graph=create_graph, retain_graph=retain_graph, allow_unused=allow_unused)


@torch.no_grad
def hvp_fd_central(closure, params: Iterable[torch.Tensor], vec: Iterable[torch.Tensor], h=1e-3):
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

    vec_h = torch._foreach_mul(vec, h)
    torch._foreach_add_(params, vec_h)
    with torch.enable_grad(): closure()
    g_plus = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    torch._foreach_sub_(params, vec_h)
    torch._foreach_sub_(params, vec_h)
    with torch.enable_grad(): closure()
    g_minus = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    torch._foreach_add_(params, vec_h)
    for p in params: p.grad = None

    hvp_ = g_plus
    torch._foreach_sub_(hvp_, g_minus)
    torch._foreach_div_(hvp_, 2*h)
    return hvp_

@torch.no_grad
def hvp_fd_forward(closure, params: Iterable[torch.Tensor], vec: Iterable[torch.Tensor], h=1e-3, g_0 = None):
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
    vec_h = torch._foreach_mul(vec, h)

    if g_0 is None:
        with torch.enable_grad(): closure()
        g_0 = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    torch._foreach_add_(params, vec_h)
    with torch.enable_grad(): closure()
    g_plus = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    torch._foreach_sub_(params, vec_h)
    for p in params: p.grad = None

    hvp_ = g_plus
    torch._foreach_sub_(hvp_, g_0)
    torch._foreach_div_(hvp_, h)
    return hvp_