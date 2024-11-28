from collections import abc

import torch

def _jacobian(input: abc.Sequence[torch.Tensor], wrt: abc.Sequence[torch.Tensor], create_graph=False):
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




def _jacobian_batched(input: abc.Sequence[torch.Tensor], wrt: abc.Sequence[torch.Tensor], create_graph=False):
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

def jacobian(input: abc.Sequence[torch.Tensor], wrt: abc.Sequence[torch.Tensor], create_graph=False, batched=True):
    if batched: return _jacobian_batched(input, wrt, create_graph)
    return _jacobian(input, wrt, create_graph)

def hessian(input: abc.Sequence[torch.Tensor], wrt: abc.Sequence[torch.Tensor], batched=True):
    return jacobian(jacobian(input, wrt, create_graph=True, batched=batched), wrt, batched=batched)

def jacobian_and_hessian(input: abc.Sequence[torch.Tensor], wrt: abc.Sequence[torch.Tensor], batched=True):
    jac = jacobian(input, wrt, create_graph=True, batched = batched)
    return jac, jacobian(jac, wrt, batched = batched)

def jacobian_list_to_vec(jacobians: abc.Sequence[torch.Tensor]):
    return torch.cat([i.ravel() for i in jacobians], 0)

def hessian_list_to_mat(hessians: abc.Sequence[torch.Tensor]):
    return torch.cat([h.reshape(h.size(0), h[1].numel()) for h in hessians], 1)