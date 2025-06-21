from contextlib import nullcontext
import torch
from ...utils import TensorList, NumberList
from ...core import Module


class SAM(Module):
    """Sharpness-Aware Minimization from https://arxiv.org/pdf/2010.01412

    SAM functions by seeking parameters that lie in neighborhoods having uniformly low loss value.
    It performs two forward and backward passes per step.

    This implementation modifies the closure to return loss and calculate gradients
    of the SAM objective. All modules after this will use the modified objective.

    .. note::
        This module requires a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients at two points on each step.

    Args:
        rho (float, optional): Neighborhood size. Defaults to 0.05.
        p (float, optional): norm of the SAM objective. Defaults to 2.
        asam (bool, optional):
            enables ASAM variant which makes perturbation relative to weight magnitudes.
            ASAM requires a much larger :code:`rho`, like 0.5 or 1.
            The :code:`tz.m.ASAM` class is idential to setting this argument to True, but
            it has larger :code:`rho` by default.

    Examples:
        SAM-SGD:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.SAM(),
                tz.m.LR(1e-2)
            )

        SAM-Adam:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.SAM(),
                tz.m.Adam(),
                tz.m.LR(1e-2)
            )

    References:
        Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2020). Sharpness-aware minimization for efficiently improving generalization. arXiv preprint arXiv:2010.01412. https://arxiv.org/abs/2010.01412#page=3.16
    """
    def __init__(self, rho: float = 0.05, p: float = 2, eps=1e-10, asam=False):
        defaults = dict(rho=rho, p=p, eps=eps, asam=asam)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):

        params = var.params
        closure = var.closure
        zero_grad = var.zero_grad
        if closure is None: raise RuntimeError("SAM requires a closure passed to the optimizer step")
        p, rho = self.get_settings(var.params, 'p', 'rho', cls=NumberList)
        s = self.settings[var.params[0]]
        eps = s['eps']
        asam = s['asam']

        # 1/p + 1/q = 1
        # okay, authors of SAM paper, I will manually solve your equation
        # so q = -p/(1-p)
        q = -p / (1-p)
        # as a validation for 2 it is -2 / -1 = 2

        @torch.no_grad
        def sam_closure(backward=True):
            orig_grads = None
            if not backward:
                # if backward is False, make sure this doesn't modify gradients
                # to avoid issues
                orig_grads = [p.grad for p in params]

            # gradient at initial parameters
            zero_grad()
            with torch.enable_grad():
                closure()

            grad = TensorList(p.grad if p.grad is not None else torch.zeros_like(p) for p in params)
            grad_abs = grad.abs()

            # compute e
            term1 = grad.sign().mul_(rho)
            term2 = grad_abs.pow(q-1)

            if asam:
                grad_abs.mul_(torch._foreach_abs(params))

            denom = grad_abs.pow_(q).sum().pow(1/p)

            e = term1.mul_(term2).div_(denom.clip(min=eps))

            if asam:
                e.mul_(torch._foreach_pow(params, 2))

            # calculate loss and gradient approximation of inner problem
            torch._foreach_add_(params, e)
            if backward:
                zero_grad()
                with torch.enable_grad():
                    # this sets .grad attributes
                    sam_loss = closure()

            else:
                sam_loss = closure(False)

            # and restore initial parameters
            torch._foreach_sub_(params, e)

            if orig_grads is not None:
                for param,orig_grad in zip(params, orig_grads):
                    param.grad = orig_grad

            return sam_loss

        var.closure = sam_closure
        return var

# different class because defaults for SAM are bad for ASAM
class ASAM(SAM):
    """Adaptive Sharpness-Aware Minimization from https://arxiv.org/pdf/2102.11600#page=6.52

    SAM functions by seeking parameters that lie in neighborhoods having uniformly low loss value.
    It performs two forward and backward passes per step.

    This implementation modifies the closure to return loss and calculate gradients
    of the SAM objective. All modules after this will use the modified objective.

    .. note::
        This module requires a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients at two points on each step.

    Args:
        rho (float, optional): Neighborhood size. Defaults to 0.05.
        p (float, optional): norm of the SAM objective. Defaults to 2.

    Examples:
        ASAM-Adam:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.ASAM(),
                tz.m.Adam(),
                tz.m.LR(1e-2)
            )

    References:
        Kwon, J., Kim, J., Park, H., & Choi, I. K. (2021, July). Asam: Adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks. In International Conference on Machine Learning (pp. 5905-5914). PMLR. https://arxiv.org/abs/2102.11600
    """
    def __init__(self, rho: float = 0.5, p: float = 2, eps=1e-10):
        super().__init__(rho=rho, p=p, eps=eps, asam=True)
