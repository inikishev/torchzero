from typing import Literal

import torch

Distributions = Literal['normal', 'uniform', 'sphere', 'rademacher']

def rademacher(shape, p: float=0.5, device=None, requires_grad = False, dtype=None, generator=None):
    """Returns a tensor filled with random numbers from Rademacher distribution.

    *p* chance to draw a -1 and 1-*p* chance to draw a 1. Looks like this:

    ```
    [-1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1,  1, -1,  1, -1, -1,  1, -1,  1]
    ```
    """
    if isinstance(shape, int): shape = (shape, )
    return torch.bernoulli(torch.full(shape, p, dtype=dtype, device=device, requires_grad=requires_grad), generator=generator) * 2 - 1

def randmask(shape, p: float=0.5, device=None, requires_grad = False, generator=None):
    """Returns a tensor randomly filled with True and False.

    *p* chance to draw `True` and 1-*p* to draw `False`."""
    return torch.rand(shape, device=device, requires_grad=requires_grad, generator=generator) < p

def uniform(shape, low: float, high: float, device=None, requires_grad=None, dtype=None):
    """Returns a tensor filled with random numbers from a uniform distribution between `low` and `high`."""
    return torch.empty(shape, device=device, dtype=dtype, requires_grad=requires_grad).uniform_(low, high)

def sphere(shape, radius: float, device=None, requires_grad=None, dtype=None, generator = None):
    """Returns a tensor filled with random numbers sampled on a unit sphere with center at 0."""
    r = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad, generator=generator)
    return (r / torch.linalg.vector_norm(r)) * radius # pylint:disable=not-callable

def sample(shape, eps: float = 1, distribution: Distributions = 'normal', generator=None, device=None, dtype=None, requires_grad=False):
    """generic random sampling function for different distributions."""
    if distribution == 'normal': return torch.randn(shape,dtype=dtype,device=device,requires_grad=requires_grad, generator=generator) * eps
    if distribution == 'uniform':
        return torch.empty(size=shape,dtype=dtype,device=device,requires_grad=requires_grad).uniform_(-eps/2, eps/2, generator=generator)

    if distribution == 'sphere': return sphere(shape, eps,dtype=dtype,device=device,requires_grad=requires_grad, generator=generator)
    if distribution == 'rademacher':
        return rademacher(shape, eps,dtype=dtype,device=device,requires_grad=requires_grad, generator=generator) * eps
    raise ValueError(f'Unknow distribution {distribution}')

def sample_like(x: torch.Tensor, eps: float = 1, distribution: Distributions = 'normal', generator=None):
    return sample(x, eps=eps, distribution=distribution, generator=generator, device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)
