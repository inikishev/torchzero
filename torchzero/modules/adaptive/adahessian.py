import math
from typing import Literal

import torch

from ...core import Chainable, Module, Target, Transform, apply_transform
from ...utils import NumberList, TensorList, as_tensorlist
from ..functional import debiased_step_size

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
        D_exp_avg_sq_.mul_(beta2).addcmul_(D, D, 1-beta2)

    else:
        assert D is None


    denom = D_exp_avg_sq_.sqrt().pow_(hessian_power).add_(eps)
    num = exp_avg_ * debiased_step_size(step+1, beta1, beta2)

    return num.div_(denom)


class AdaHessian(Module):
    """AdaHessian: An Adaptive Second Order Optimizer for Machine Learning (https://arxiv.org/abs/2006.00719)

    This is similar to Adam, but the second momentum is replaced by square root of an exponential moving average of random hessian-vector products.

    Notes:
        - In most cases AdaHessian should be the first module in the chain because it relies on autograd. Use the ``inner`` argument if you wish to apply AdaHessian preconditioning to another module's output.

        - If you are using gradient estimators or reformulations, set ``hvp_method`` to "forward" or "central".

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
            Determines how Hessian-vector products are evaluated.

            - ``"autograd"``: Use PyTorch's autograd to calculate exact HVPs.
              This requires creating a graph for the gradient.
            - ``"forward"``: Use a forward finite difference formula to
              approximate the HVP. This requires one extra gradient evaluation.
            - ``"central"``: Use a central finite difference formula for a
              more accurate HVP approximation. This requires two extra
              gradient evaluations.
            Defaults to "autograd".
        fd_h (float, optional): finite difference step size if ``hvp_method`` is "forward" or "central". Defaults to 1e-3.
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
        hvp_method: Literal['autograd', 'forward', 'central'] = 'autograd',
        fd_h: float = 1e-3,
        n_samples = 1,
        seed: int | None = None,
        inner: Chainable | None = None
    ):
        defaults = dict(beta1=beta1, beta2=beta2, update_freq=update_freq, averaging=averaging, block_size=block_size, eps=eps, hessian_power=hessian_power, hvp_method=hvp_method, n_samples=n_samples, fd_h=fd_h, seed=seed)
        super().__init__(defaults)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def step(self, var):
        params = var.params
        settings = self.settings[params[0]]
        hvp_method = settings['hvp_method']
        fd_h = settings['fd_h']
        update_freq = settings['update_freq']
        n_samples = settings['n_samples']

        seed = settings['seed']
        generator = self.get_generator(params[0].device, seed)

        beta1, beta2, eps, averaging, block_size, hessian_power = self.get_settings(params,
            'beta1', 'beta2', 'eps', 'averaging', 'block_size', "hessian_power", cls=NumberList)

        exp_avg, D_exp_avg_sq = self.get_state(params, 'exp_avg', 'h_exp_avg', cls=TensorList)

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        closure = var.closure
        assert closure is not None

        D = None
        if step % update_freq == 0:

            rgrad=None
            for i in range(n_samples):
                u = [_rademacher_like(p, generator=generator) for p in params]

                Hvp, rgrad = self.Hvp(u, at_x0=True, var=var, rgrad=rgrad, hvp_method=hvp_method,
                                     h=fd_h, normalize=True, retain_grad=i < n_samples-1)
                Hvp = tuple(Hvp)

                if D is None: D = Hvp
                else: torch._foreach_add_(D, Hvp)

            assert D is not None
            if n_samples > 1: torch._foreach_div_(D, n_samples)

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
