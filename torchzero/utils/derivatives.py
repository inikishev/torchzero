from collections.abc import Iterable, Sequence

import torch
import torch.autograd.forward_ad as fwAD

from .torch_tools import swap_tensors_no_use_count_check, vec_to_tensors
from .tensorlist import TensorList

def _jacobian(outputs: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False):
    flat_outputs = torch.cat([i.ravel() for i in outputs])
    grad_ouputs = torch.eye(len(flat_outputs), device=outputs[0].device, dtype=outputs[0].dtype)
    jac = []
    for i in range(flat_outputs.numel()):
        jac.append(torch.autograd.grad(
            flat_outputs,
            wrt,
            grad_ouputs[i],
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
            is_grads_batched=False,
        ))
    return [torch.stack(z) for z in zip(*jac)]


def _jacobian_batched(outputs: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False):
    flat_outputs = torch.cat([i.ravel() for i in outputs])
    return torch.autograd.grad(
        flat_outputs,
        wrt,
        torch.eye(len(flat_outputs), device=outputs[0].device, dtype=outputs[0].dtype),
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=True,
        is_grads_batched=True,
    )

def flatten_jacobian(jacs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Converts the output of jacobian_wrt (a list of tensors) into a single 2D matrix.

    Args:
        jacs (Sequence[torch.Tensor]):
            output from jacobian_wrt where ach tensor has the shape ``(*output.shape, *wrt[i].shape)``.

    Returns:
        torch.Tensor: has the shape ``(output.ndim, wrt.ndim)``.
    """
    if not jacs:
        return torch.empty(0, 0)

    n_out = jacs[0].shape[0]
    return torch.cat([j.reshape(n_out, -1) for j in jacs], dim=1)


def jacobian_wrt(outputs: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False, batched=True) -> Sequence[torch.Tensor]:
    """Calculate jacobian of a sequence of tensors w.r.t another sequence of tensors.
    Returns a sequence of tensors with the length as `wrt`.
    Each tensor will have the shape `(*output.shape, *wrt[i].shape)`.

    Args:
        outputs (Sequence[torch.Tensor]): input sequence of tensors.
        wrt (Sequence[torch.Tensor]): sequence of tensors to differentiate w.r.t.
        create_graph (bool, optional):
            pytorch option, if True, graph of the derivative will be constructed,
            allowing to compute higher order derivative products. Default: False.
        batched (bool, optional): use faster but experimental pytorch batched jacobian
            This only has effect when `input` has more than 1 element. Defaults to True.

    Returns:
        sequence of tensors with the length as `wrt`.
    """
    if batched: return _jacobian_batched(outputs, wrt, create_graph)
    return _jacobian(outputs, wrt, create_graph)

def jacobian_and_hessian_wrt(outputs: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False, batched=True):
    """Calculate jacobian and hessian of a sequence of tensors w.r.t another sequence of tensors.
    Calculating hessian requires calculating the jacobian. So this function is more efficient than
    calling `jacobian` and `hessian` separately, which would calculate jacobian twice.

    Args:
        outputs (Sequence[torch.Tensor]): input sequence of tensors.
        wrt (Sequence[torch.Tensor]): sequence of tensors to differentiate w.r.t.
        create_graph (bool, optional):
            pytorch option, if True, graph of the derivative will be constructed,
            allowing to compute higher order derivative products. Default: False.
        batched (bool, optional): use faster but experimental pytorch batched grad. Defaults to True.

    Returns:
        tuple with jacobians sequence and hessians sequence.
    """
    jac = jacobian_wrt(outputs, wrt, create_graph=True, batched = batched)
    return jac, jacobian_wrt(jac, wrt, batched = batched, create_graph=create_graph)


# def hessian_list_to_mat(hessians: Sequence[torch.Tensor]):
#     """takes output of `hessian` and returns the 2D hessian matrix.
#     Note - I only tested this for cases where input is a scalar."""
#     return torch.cat([h.reshape(h.size(0), h[1].numel()) for h in hessians], 1)

def jacobian_and_hessian_mat_wrt(outputs: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False, batched=True):
    """Calculate jacobian and hessian of a sequence of tensors w.r.t another sequence of tensors.
    Calculating hessian requires calculating the jacobian. So this function is more efficient than
    calling `jacobian` and `hessian` separately, which would calculate jacobian twice.

    Args:
        outputs (Sequence[torch.Tensor]): input sequence of tensors.
        wrt (Sequence[torch.Tensor]): sequence of tensors to differentiate w.r.t.
        create_graph (bool, optional):
            pytorch option, if True, graph of the derivative will be constructed,
            allowing to compute higher order derivative products. Default: False.
        batched (bool, optional): use faster but experimental pytorch batched grad. Defaults to True.

    Returns:
        tuple with jacobians sequence and hessians sequence.
    """
    jac = jacobian_wrt(outputs, wrt, create_graph=True, batched = batched)
    H_list = jacobian_wrt(jac, wrt, batched = batched, create_graph=create_graph)
    return flatten_jacobian(jac), flatten_jacobian(H_list)

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
    ```python
    model = nn.Linear(4, 2) # (2, 4) weight and (2, ) bias
    X = torch.randn(10, 4)
    y = torch.randn(10, 2)

    def fn():
        y_hat = model(X)
        loss = F.mse_loss(y_hat, y)
        return loss

    hessian_mat(fn, model.parameters()) # list of two lists of two lists of 3D and 4D tensors
    ```

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
    ```python
    model = nn.Linear(4, 2) # 10 parameters in total
    X = torch.randn(10, 4)
    y = torch.randn(10, 2)

    def fn():
        y_hat = model(X)
        loss = F.mse_loss(y_hat, y)
        return loss

    hessian_mat(fn, model.parameters()) # 10x10 tensor
    ```

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
    ```python
    model = nn.Linear(4, 2)
    X = torch.randn(10, 4)
    y = torch.randn(10, 2)

    tangent = [torch.randn_like(p) for p in model.parameters()]

    def fn():
        y_hat = model(X)
        loss = F.mse_loss(y_hat, y)
        return loss

    jvp(fn, model.parameters(), tangent) # scalar
    ```
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
    normalize=True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Jacobian vector product using central finite difference formula.

    Example:
    ```python
    model = nn.Linear(4, 2)
    X = torch.randn(10, 4)
    y = torch.randn(10, 2)

    tangent = [torch.randn_like(p) for p in model.parameters()]

    def fn():
        y_hat = model(X)
        loss = F.mse_loss(y_hat, y)
        return loss

    jvp_fd_central(fn, model.parameters(), tangent) # scalar
    ```
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
    normalize=True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Jacobian vector product using forward finite difference formula.
    Loss at initial point can be specified in the `v_0` argument.

    Example:
    ```python
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
    ```

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


@torch.no_grad
def hvp_fd_central(
    closure,
    params: Iterable[torch.Tensor],
    x: Iterable[torch.Tensor],
    h=1e-3,
    normalize=True,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    """Returns ``(loss_approx, Hx)``.

    Please note that this will clear ``grad`` attributes in params.

    Example:
    ```python
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
    ```
    """
    params = list(params)
    x = list(x)

    vec_norm = None
    if normalize:
        vec_norm = torch.linalg.vector_norm(torch.cat([t.view(-1) for t in x])) # pylint:disable=not-callable
        if vec_norm == 0: return None, [torch.zeros_like(p) for p in params]
        x = torch._foreach_div(x, vec_norm)

    xh = torch._foreach_mul(x, h)
    torch._foreach_add_(params, xh)
    with torch.enable_grad(): loss = closure()
    g_plus = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    torch._foreach_sub_(params, xh)
    torch._foreach_sub_(params, xh)
    with torch.enable_grad(): loss = closure()
    g_minus = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    torch._foreach_add_(params, xh)
    for p in params: p.grad = None

    hx = g_plus
    torch._foreach_sub_(hx, g_minus)
    torch._foreach_div_(hx, 2*h)

    if normalize: torch._foreach_mul_(hx, vec_norm)
    return loss, hx

@torch.no_grad
def hvp_fd_forward(
    closure,
    params: Iterable[torch.Tensor],
    x: Iterable[torch.Tensor],
    h=1e-3,
    g_0=None,
    normalize=True,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    """Returns ``(loss_approx, Hx)``.

    Gradient at initial point can be specified in the ``g_0`` argument.

    Please note that this will clear ``grad`` attributes in params.

    Example:
    ```python
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
    ```
    """

    params = list(params)
    x = list(x)
    loss = None

    vec_norm = None
    if normalize:
        vec_norm = torch.linalg.vector_norm(torch.cat([t.ravel() for t in x])) # pylint:disable=not-callable
        if vec_norm == 0: return None, [torch.zeros_like(p) for p in params]
        x = torch._foreach_div(x, vec_norm)

    xh = torch._foreach_mul(x, h)

    if g_0 is None:
        with torch.enable_grad(): loss = closure()
        g_0 = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    else:
        g_0 = list(g_0)

    torch._foreach_add_(params, xh)
    with torch.enable_grad():
        l = closure()
        if loss is None: loss = l
    g_plus = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    torch._foreach_sub_(params, xh)
    for p in params: p.grad = None

    hx = g_plus
    torch._foreach_sub_(hx, g_0)
    torch._foreach_div_(hx, h)

    if normalize: torch._foreach_mul_(hx, vec_norm)
    return loss, hx

@torch.no_grad
def hessian_fd(fn, params: Sequence[torch.Tensor], eps: float = 1e-4, full: bool = True):
    """returns ``f(x), g(x), H(x)``, where ``g(x)`` is a tensor list.

    Number of evals for full is: 4n^2 - 2n + 1

    Number of evals for upper is: 2n^2 + 1.
    """
    params = TensorList(params)
    p_0 = params.clone()
    n = sum(t.numel() for t in params)
    device = params[0].device; dtype = params[0].dtype
    fx = fn()
    g = params.zeros_like()
    H = torch.zeros((n, n), device=device, dtype=dtype)

    for i in range(n):
        for j in (range(n) if full else range(i, n)):
            if i == j:
                params.flat_set_lambda_(i, lambda x: x + eps)
                f_plus = fn()

                params.flat_set_lambda_(i, lambda x: x - 2 * eps)
                f_minus = fn()

                # params.flat_set_lambda_(i, lambda x: x + eps)
                g.flat_set_(i, (f_plus - f_minus) / (2*eps))
                H[i, i] = (f_plus - 2 * fx + f_minus) / (eps ** 2)

            else:
                params.flat_set_lambda_(i, lambda x: x + eps)
                params.flat_set_lambda_(j, lambda x: x + eps)
                f_pp = fn()

                params.flat_set_lambda_(i, lambda x: x - 2 * eps)
                f_np = fn()

                params.flat_set_lambda_(j, lambda x: x - 2 * eps)
                f_nn = fn()

                params.flat_set_lambda_(i, lambda x: x + 2 * eps)
                f_pn = fn()

                # params.flat_set_lambda_(i, lambda x: x - eps)
                # params.flat_set_lambda_(j, lambda x: x + eps)

                H[i, j] = (f_pp - f_np - f_pn + f_nn) / (4 * eps ** 2)
                if not full: H[j, i] = H[i, j]

            params.copy_(p_0) # otherwise inaccuracy builds up

    if full:
        H = H + H.T
        H /= 2

    return fx, g, H