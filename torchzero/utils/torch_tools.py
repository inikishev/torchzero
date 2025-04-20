from collections.abc import Iterable
from typing import Any

import torch, numpy as np

def totensor(x):
    if isinstance(x, torch.Tensor): return x
    if isinstance(x, np.ndarray): return torch.from_numpy(x)
    return torch.from_numpy(np.asarray(x))

def tonumpy(x):
    if isinstance(x, np.ndarray): return x
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x)

def tofloat(x) -> float:
    if isinstance(x, float): return x
    if isinstance(x, torch.Tensor): return x.detach().cpu().item()
    if isinstance(x, np.ndarray): return x.item() # type:ignore
    return float(x)

def tolist(x):
    if isinstance(x, list): return x
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray): return x.tolist()
    return np.asarray(x).tolist()

def vec_to_tensors(vec: torch.Tensor, reference: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    tensors = []
    cur = 0
    for r in reference:
        numel = r.numel()
        tensors.append(vec[cur:cur+numel].view_as(r))
        cur += numel
    return tensors

def vec_to_tensors_(vec: torch.Tensor, tensors_: Iterable[torch.Tensor]):
    cur = 0
    for t in tensors_:
        numel = t.numel()
        t.set_(vec[cur:cur+numel].view_as(t)) # pyright: ignore[reportArgumentType]
        cur += numel

