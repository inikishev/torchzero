from collections.abc import Iterable
from typing import Literal

import torch
import torch.autograd.forward_ad as fwAD

from ...core import OptimizerModule, _ClosureType
from ...tensorlist import TensorList
from ...random import Distributions
from ...utils.torch_tools import swap_tensors_no_use_count_check
from .base_approximator import GradientApproximatorBase

def get_forward_gradient(
    params: Iterable[torch.Tensor],
    closure: _ClosureType,
    n_samples: int,
    distribution: Distributions,
    mode: Literal["jvp", "grad", "fd"],
    fd_eps: float = 1e-4,
):
    """Evaluates forward gradient of a closure w.r.t iterable of parameters with a random tangent vector.

    Args:
        params (Iterable[torch.Tensor]): iterable of parameters of the model.
        closure (_ClosureType):
            A closure that reevaluates the model and returns the loss.
            Closure must accept `backward = True` boolean argument. Forward gradient will always call it as
            `closure(False)`, unless `mode = "grad"` which requires a backward pass.
        n_samples (int): number of forward gradients to evaluate and average.
        distribution (Distributions): distribution for random tangent vector.
        mode (str):
            "jvp" - uses forward mode AD, usually slightly slower than backward mode AD  but uses significantly less memory.

            "grad" - evaluates gradient with `loss.backward()` which may be faster but uses all the memory, mainly useful for
            benchmarking as there is probably no point in forward gradient if full gradient is available.

            "fd" - uses finite difference to estimate JVP in two forward passes,
            doesn't require the objective to be autodiffable. Equivalent to randomized FDM.

        fd_eps (float, optional): epsilon for finite difference, only has effect if mode is "fd". Defaults to 1e-4.

    Returns:
        TensorList: list of estimated gradients of the same structure and shape as `params`.

    Reference:
        Baydin, A. G., Pearlmutter, B. A., Syme, D., Wood, F., & Torr, P. (2022).
        Gradients without backpropagation. arXiv preprint arXiv:2202.08587.
        https://arxiv.org/abs/2202.08587
    """
    if not isinstance(params, TensorList): params = TensorList(params)
    params = params.with_requires_grad()

    orig_params = None
    grad = None
    loss = None
    for _ in range(n_samples):

        # generate random vector
        tangents = params.sample_like(fd_eps if mode == 'fd' else 1, distribution)

        if mode == 'jvp':
            if orig_params is None:
                orig_params = params.clone().requires_grad_()

            # evaluate jvp with it
            with fwAD.dual_level():

                # swap to duals
                for param, clone, tangent in zip(params, orig_params, tangents):
                    dual = fwAD.make_dual(clone, tangent)
                    torch.utils.swap_tensors(param, dual)

                loss = closure(False)
                jvp = fwAD.unpack_dual(loss).tangent

        elif mode == 'grad':
            with torch.enable_grad(): loss = closure()
            jvp = tangents.mul(params.ensure_grad_().grad).sum()

        elif mode == 'fd':
            loss = closure(False)
            params += tangents
            loss2 = closure(False)
            params -= tangents
            jvp = (loss2 - loss) / fd_eps**2

        else:
            raise ValueError(mode)

        # update grad estimate
        if grad is None: grad = tangents * jvp
        else: grad += tangents * jvp

    # swap back to original params
    if orig_params is not None:
        for param, orig in zip(params, orig_params):
            swap_tensors_no_use_count_check(param, orig)

    assert grad is not None
    assert loss is not None
    if n_samples > 1:
        grad /= n_samples

    return grad, loss

class ForwardGradient(GradientApproximatorBase):
    """Evaluates jacobian-vector product with a random vector using forward mode autodiff (torch.autograd.forward_ad), which is
    the true directional derivative in the direction of that vector.

    Args:
        n_samples (int): number of forward gradients to evaluate and average.
        distribution (Distributions): distribution for random tangent vector.
        mode (str):
            "jvp" - uses forward mode AD, usually slightly slower than backward mode AD but uses significantly less memory,
            because it doesn't have to store intermediate activations.

            "grad" - evaluates gradient with `loss.backward()` which may be faster but uses all the memory, mainly useful for
            benchmarking as there is probably no point in forward gradient if full gradient is available.

            "fd" - uses finite difference to estimate JVP in two forward passes,
            doesn't require the objective to be autodiffable. Equivalent to randomized FDM.

        fd_eps (float, optional): epsilon for finite difference, only has effect if mode is "fd". Defaults to 1e-4.
        target (str, optional):
            determines what this module sets.

            "ascent" - it creates a new ascent direction but doesn't treat is as gradient.

            "grad" - it creates the gradient and sets it to `.grad` attributes (default).

            "closure" - it makes a new closure that sets the estimated gradient to the `.grad` attributes.

    Reference:
        Baydin, A. G., Pearlmutter, B. A., Syme, D., Wood, F., & Torr, P. (2022).
        Gradients without backpropagation. arXiv preprint arXiv:2202.08587.
        https://arxiv.org/abs/2202.08587
    """
    def __init__(
        self,
        n_samples: int = 1,
        distribution: Distributions = "normal",
        mode: Literal["jvp", "grad", "fd"] = "jvp",
        fd_eps: float = 1e-4,
        target: Literal['ascent', 'grad', 'closure'] = 'grad',
    ):
        super().__init__({}, requires_fx0=False, target = target)
        self.distribution: Distributions = distribution
        self.n_samples = n_samples
        self.mode: Literal["jvp", "grad", "fd"] = mode
        self.fd_eps = fd_eps


    def _make_ascent(self, closure, params, fx0):
        g, fx0 = get_forward_gradient(
            params=params,
            closure=closure,
            n_samples=self.n_samples,
            distribution=self.distribution,
            mode=self.mode,
            fd_eps=self.fd_eps,
        )

        return g, fx0, None