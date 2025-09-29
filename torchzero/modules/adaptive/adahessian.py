import math
from typing import Literal

import torch

from ...core import Chainable, Transform, HVPMethod
from ...utils import NumberList, TensorList, Distributions, unpack_dicts, unpack_states

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


class AdaHessian(Transform):
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
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.AdaHessian(),
        tz.m.LR(0.1)
    )
    ```

    AdaHessian preconditioner can be applied to any other module by passing it to the ``inner`` argument.
    Turn off AdaHessian's first momentum to get just the preconditioning. Here is an example of applying
    AdaHessian preconditioning to nesterov momentum (``tz.m.NAG``):
    ```python
    opt = tz.Optimizer(
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
        debias: bool = True,
        seed: int | None = None,

        exp_avg_tfm: Chainable | None = None,
        D_exp_avg_sq_tfm: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults["exp_avg_tfm"], defaults["D_exp_avg_sq_tfm"]
        super().__init__(defaults)

        self.set_child('exp_avg', exp_avg_tfm)
        self.set_child('D_exp_avg_sq', D_exp_avg_sq_tfm)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        params = objective.params

        beta1, beta2, averaging, block_size = unpack_dicts(settings, 'beta1', 'beta2', 'averaging', 'block_size', cls=NumberList)

        exp_avg, D_exp_avg_sq = unpack_states(states, params, 'exp_avg', 'D_exp_avg_sq', cls=TensorList)

        # ---------------------------- hutchinson hessian ---------------------------- #
        fs = settings[0]
        step = self.increment_counter("step", start=0) # 0 on 1st update
        update_freq = fs['update_freq']

        if step % update_freq == 0:
            self.increment_counter("num_Ds", start=1)

            D, _ = objective.hutchinson_hessian(
                rgrad = None,
                at_x0 = True,
                n_samples = fs['n_samples'],
                distribution = fs['distribution'],
                hvp_method = fs['hvp_method'],
                h = fs['h'],
                zHz = fs["zHz"],
                generator = self.get_generator(params[0].device, fs["seed"]),
            )

            D = TensorList(D).zipmap_args(_block_average, block_size, averaging)
            D_exp_avg_sq.mul_(beta2).addcmul_(D, D, value=1-beta2)

        # --------------------------------- momentum --------------------------------- #
        tensors = objective.get_updates() # do this after hutchinson to not disturb autograd
        exp_avg.lerp_(tensors, 1-beta1)


    @torch.no_grad
    def apply_states(self, objective, states, settings):
        params = objective.params

        beta1, beta2, eps, hessian_power = unpack_dicts(settings, 'beta1', 'beta2', 'eps', 'hessian_power', cls=NumberList)
        exp_avg, D_exp_avg_sq = unpack_states(states, params, 'exp_avg', 'D_exp_avg_sq', cls=TensorList)

        # ---------------------------------- debias ---------------------------------- #
        if settings[0]["debias"]:
            bias_correction1 = 1.0 - (beta1 ** (self.global_state["step"] + 1))
            bias_correction2 = 1.0 - (beta2 ** self.global_state["num_Ds"])
            exp_avg = exp_avg / bias_correction1
            D_exp_avg_sq = D_exp_avg_sq / bias_correction2


        # -------------------------------- transforms -------------------------------- #
        exp_avg = TensorList(self.inner_step_tensors(
            "exp_avg", tensors=exp_avg, clone=True, objective=objective, must_exist=False))

        D_exp_avg_sq = TensorList(self.inner_step_tensors(
            "D_exp_avg_sq", tensors=D_exp_avg_sq, clone=True, objective=objective, must_exist=False))

        # ------------------------------ compute update ------------------------------ #
        denom = D_exp_avg_sq.lazy_pow(hessian_power / 2) + eps
        objective.updates = exp_avg / denom
        return objective