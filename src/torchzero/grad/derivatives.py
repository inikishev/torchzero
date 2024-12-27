from collections.abc import Sequence, Iterable

import torch

def _jacobian(input: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False):
    flat_input = torch.cat([i.reshape(-1) for i in input])
    grad_ouputs = torch.eye(len(flat_input), device=input[0].device, dtype=input[0].dtype)
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



def _jacobian_batched(input: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False):
    flat_input = torch.cat([i.reshape(-1) for i in input])
    return torch.autograd.grad(
        flat_input,
        wrt,
        torch.eye(len(flat_input), device=input[0].device, dtype=input[0].dtype),
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=True,
        is_grads_batched=True,
    )

def jacobian(input: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False, batched=True) -> Sequence[torch.Tensor]:
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
    if batched: return _jacobian_batched(input, wrt, create_graph)
    return _jacobian(input, wrt, create_graph)

def hessian(input: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False, batched=True):
    """Calculate hessian of a sequence of tensors w.r.t another sequence of tensors.
    Returns a sequence of tensors with the length as `wrt`.
    If you need a hessian matrix out of that sequence, pass it to `hessian_list_to_mat`.

    Args:
        input (Sequence[torch.Tensor]): input sequence of tensors.
        wrt (Sequence[torch.Tensor]): sequence of tensors to differentiate w.r.t.
        create_graph (bool, optional):
            pytorch option, if True, graph of the derivative will be constructed,
            allowing to compute higher order derivative products. Default: False.
        batched (bool, optional): use faster but experimental pytorch batched grad. Defaults to True.

    Returns:
        sequence of tensors with the length as `wrt`.
    """
    return jacobian(jacobian(input, wrt, create_graph=True, batched=batched), wrt, create_graph=create_graph, batched=batched)

def jacobian_and_hessian(input: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False, batched=True):
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
    jac = jacobian(input, wrt, create_graph=True, batched = batched)
    return jac, jacobian(jac, wrt, batched = batched, create_graph=create_graph)

def jacobian_list_to_vec(jacobians: Iterable[torch.Tensor]):
    """flattens and concatenates a sequence of tensors."""
    return torch.cat([i.ravel() for i in jacobians], 0)

def hessian_list_to_mat(hessians: Sequence[torch.Tensor]):
    """takes output of `hessian` and returns the 2D hessian matrix.
    Note - I only tested this for cases where input is a scalar."""
    return torch.cat([h.reshape(h.size(0), h[1].numel()) for h in hessians], 1)