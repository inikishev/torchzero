import torch

def rademacher(shape, p=0.5, device=None, requires_grad = False, dtype=None):
    """100p% to draw a -1 and 100(1-p)% to draw a 1. Looks like this:
    ```
    [-1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1,  1, -1,  1, -1, -1,  1, -1,  1]
    ```
    """
    if isinstance(shape, int): shape = (shape, )
    return torch.bernoulli(torch.full(shape, p, dtype=dtype, device=device, requires_grad=requires_grad)) * 2 - 1

def randmask(shape, p=0.5, device=None, requires_grad = False):
    """100p% chance to draw True and 100(1-p)% to draw False."""
    return torch.rand(shape, device=device, requires_grad=requires_grad) < p

def uniform(shape, low, high, device=None, requires_grad=None, dtype=None):
    """Returns a tensor filled with random numbers from a uniform distribution."""
    return torch.empty(shape, device=device, dtype=dtype, requires_grad=requires_grad).uniform_(low, high)

def sphere(shape, radius, device=None, requires_grad=None, dtype=None):
    """Returns a tensor filled with random numbers sampled from a unit sphere with center at 0."""
    r = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    return (r / torch.linalg.vector_norm(r)) * radius # pylint:disable=not-callable