import math
from typing import Literal

import torch

from ...core import Chainable, Module, Target, Transform, apply_transform, HVPMethod
from ...utils import NumberList, TensorList, Distributions

def _full_average(hvp: torch.Tensor):
    if hvp.ndim >= 3:  # Conv kernel
        return torch.mean(hvp.abs(), dim=[2, *range(3,hvp.ndim)], keepdim=True)
    return hvp

def _block_average(x: torch.Tensor, block_size: int | None, enable: bool):
    """averages x over first dimension in blocks"""
    if enable and x.ndim >= 2:
        if math.prod(x.shape[1:]) <= 1: return x
        if block_size is None: return _full_average(x)
        size = x.size(0)

        n_blocks = size // block_size
        if n_blocks <= 1: return x.abs().mean(0, keepdim = True)

        n_remaining = size - n_blocks * block_size
        remaining = None
        if n_remaining > 0:
            remaining = x[-n_remaining:].abs().mean(0, keepdim=True).repeat_interleave(n_remaining, 0)
            x = x[:-n_remaining]

        x = x.view(block_size, n_blocks, *x.shape[1:])
        x_mean = x.abs().mean(0).repeat_interleave(block_size, 0)

        if remaining is None: return x_mean
        return torch.cat([x_mean, remaining], 0)

    return x


def _rademacher_like(tensor, p = 0.5, generator = None):
    """p is probability of a 1, other values will be -1."""
    return torch.bernoulli(torch.full_like(tensor, p), generator = generator).mul_(2).sub_(1)

def adahessian(
    tensors: TensorList,
    D: TensorList | None,
    exp_avg_: TensorList,
    D_exp_avg_sq_: TensorList,
    beta1: float | NumberList,
    beta2: float | NumberList,
    update_freq: int,
    eps: float | NumberList,
    hessian_power: float | NumberList,
    step: int,
):
    # momentum
    exp_avg_.lerp_(tensors, 1-beta1)

    # update preconditioner
    if step % update_freq == 0:
        assert D is not None
        D_exp_avg_sq_.mul_(beta2).addcmul_(D, D, 1 - beta2)

    else:
        assert D is None

    bias_correction1 = 1.0 - (beta1 ** (step + 1))
    bias_correction2 = 1.0 - (beta2 ** (step + 1))

    denom = (D_exp_avg_sq_ / bias_correction2).pow_(hessian_power / 2).add_(eps)

    return (exp_avg_ / denom).div_(bias_correction1)


class AdaHessian(Module):
    """AdaHessian: An Adaptive Second Order Optimizer for Machine Learning (https://arxiv.org/abs/2006.00719)

    This is similar to Adam, but the second momentum is replaced by square root of an exponential moving average of random hessian-vector products.

    Notes:
        - In most cases AdaHessian should be the first module in the chain because it relies on autograd. Use the ``inner`` argument if you wish to apply AdaHessian preconditioning to another module's output.

        - This module requires a closure passed to the optimizer step, as it needs to re-evaluate the loss and gradients for calculating HVPs. The closure must accept a ``backward`` argument (refer to documentation).

    Args:
        beta1 (float, optional): first momentum. Defaults to 0.9.
        beta2 (float, optional): second momentum for squared hessian diagonal estimates. Defaults to 0.999.
        averaging (bool, optional):
            whether to enable block diagonal averaging over 1st dimension on parameters that have 2+ dimensions.
            This can be set per-parameter in param groups.
        block_size (int, optional):
            size of block in the block-diagonal averaging.
        update_freq (int, optional):
            frequency of updating hessian diagonal estimate via a hessian-vector product.
            This value can be increased to reduce computational cost. Defaults to 1.
        eps (float, optional):
            division stability epsilon. Defaults to 1e-8.
        hvp_method (str, optional):
            Determines how hessian-vector products are computed.

            - ``"batched_autograd"`` - uses autograd with batched hessian-vector products. If a single hessian-vector is evaluated, equivalent to ``"autograd"``. Faster than ``"autograd"`` but uses more memory.
            - ``"autograd"`` - uses autograd hessian-vector products. If multiple hessian-vector products are evaluated, uses a for-loop. Slower than ``"batched_autograd"`` but uses less memory.
            - ``"fd_forward"`` - uses gradient finite difference approximation with a less accurate forward formula which requires one extra gradient evaluation per hessian-vector product.
            - ``"fd_central"`` - uses gradient finite difference approximation with a more accurate central formula which requires two gradient evaluations per hessian-vector product.

            Defaults to ``"autograd"``.
        h (float, optional):
            The step size for finite difference if ``hvp_method`` is
            ``"fd_forward"`` or ``"fd_central"``. Defaults to 1e-3.
        n_samples (int, optional):
            number of hessian-vector products with random vectors to evaluate each time when updating
            the preconditioner. Larger values may lead to better hessian diagonal estimate. Defaults to 1.
        seed (int | None, optional): seed for random vectors. Defaults to None.
        inner (Chainable | None, optional):
            Inner module. If this is specified, operations are performed in the following order.
            1. compute hessian diagonal estimate.
            2. pass inputs to ``inner``.
            3. momentum and preconditioning are applied to the ouputs of ``inner``.

    ## Examples:

    Using AdaHessian:

    ```python
    opt = tz.Modular(
        model.parameters(),
        tz.m.AdaHessian(),
        tz.m.LR(0.1)
    )
    ```

    AdaHessian preconditioner can be applied to any other module by passing it to the ``inner`` argument.
    Turn off AdaHessian's first momentum to get just the preconditioning. Here is an example of applying
    AdaHessian preconditioning to nesterov momentum (``tz.m.NAG``):
    ```python
    opt = tz.Modular(
        model.parameters(),
        tz.m.AdaHessian(beta1=0, inner=tz.m.NAG(0.9)),
        tz.m.LR(0.1)
    )
    ```

    """
    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        averaging: bool = True,
        block_size: int | None = None,
        update_freq: int = 1,
        eps: float = 1e-8,
        hessian_power: float = 1,
        distribution: Distributions = 'rademacher',
        hvp_method: HVPMethod = 'autograd',
        h: float = 1e-3,
        n_samples = 1,
        zHz: bool = True,
        seed: int | None = None,
        inner: Chainable | None = None
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner']
        super().__init__(defaults)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def step(self, var):
        params = var.params

        beta1, beta2, eps, averaging, block_size, hessian_power = self.get_settings(params,
            'beta1', 'beta2', 'eps', 'averaging', 'block_size', "hessian_power", cls=NumberList)

        exp_avg, D_exp_avg_sq = self.get_state(params, 'exp_avg', 'h_exp_avg', cls=TensorList)

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        closure = var.closure
        assert closure is not None

        D = None
        update_freq = self.defaults['update_freq']
        if step % update_freq == 0:

            D, _ = var.hutchinson_hessian(
                rgrad = None,
                at_x0 = True,
                n_samples = self.defaults['n_samples'],
                distribution = self.defaults['distribution'],
                hvp_method = self.defaults['hvp_method'],
                h = self.defaults['h'],
                zHz = self.defaults["zHz"],
                generator = self.get_generator(params[0].device, self.defaults["seed"]),
            )

            D = TensorList(D).zipmap_args(_block_average, block_size, averaging)

        update = var.get_update()
        if 'inner' in self.children:
            update = apply_transform(self.children['inner'], tensors=update, params=params, grads=var.grad, var=var)

        var.update = adahessian(
            tensors=TensorList(update),
            D=TensorList(D) if D is not None else None,
            exp_avg_=exp_avg,
            D_exp_avg_sq_=D_exp_avg_sq,
            beta1=beta1,
            beta2=beta2,
            update_freq=update_freq,
            eps=eps,
            hessian_power=hessian_power,
            step=step,
        )

        return var
