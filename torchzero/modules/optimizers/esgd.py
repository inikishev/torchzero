import math
from collections.abc import Callable
from typing import Literal

import torch

from ...core import Chainable, Module, Target, Transform, apply_transform
from ...utils import NumberList, TensorList, as_tensorlist


def esgd_(
    tensors_: TensorList,
    D: TensorList | None,
    D_sq_acc_: TensorList,
    damping: float | NumberList,
    update_freq: int,
    step: int,
    i: int,
):
    # update preconditioner
    if step % update_freq == 0:
        assert D is not None
        D_sq_acc_.addcmul_(D, D)
        i += 1
    else:
        assert D is None

    denom = (D_sq_acc_ / max(i, 1)).sqrt_().add_(damping)
    return tensors_.div_(denom), i


class ESGD(Module):
    """Equilibrated Gradient Descent (https://arxiv.org/abs/1502.04390)

    This is similar to Adagrad, but the accumulates squared randomized hessian diagonal estimates instead of squared gradients.

    .. note::
        In most cases Adagrad should be the first module in the chain because it relies on autograd. Use the :code:`inner` argument if you wish to apply Adagrad preconditioning to another module's output.

    .. note::
        If you are using gradient estimators or reformulations, set :code:`hvp_method` to "forward" or "central".

    .. note::
        This module requires a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients for calculating HVPs.
        The closure must accept a ``backward`` argument (refer to documentation).

    Args:
        damping (float, optional): added to denominator for stability. Defaults to 1e-4.
        update_freq (int, optional):
            frequency of updating hessian diagonal estimate via a hessian-vector product.
            This value can be increased to reduce computational cost. Defaults to 20.
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
        h (float, optional): finite difference step size if :code:`hvp_method` is "forward" or "central". Defaults to 1e-3.
        n_samples (int, optional):
            number of hessian-vector products with random vectors to evaluate each time when updating
            the preconditioner. Larger values may lead to better hessian diagonal estimate. Defaults to 1.
        seed (int | None, optional): seed for random vectors. Defaults to None.
        inner (Chainable | None, optional):
            Inner module. If this is specified, operations are performed in the following order.
            1. compute hessian diagonal estimate.
            2. pass inputs to :code:`inner`.
            3. momentum and preconditioning are applied to the ouputs of :code:`inner`.

    Examples:
        Using ESGD:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.ESGD(),
                tz.m.LR(0.1)
            )

        ESGD preconditioner can be applied to any other module by passing it to the :code:`inner` argument. Here is an example of applying
        ESGD preconditioning to nesterov momentum (:code:`tz.m.NAG`):

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.ESGD(beta1=0, inner=tz.m.NAG(0.9)),
                tz.m.LR(0.1)
            )

    """
    def __init__(
        self,
        damping: float = 1e-4,
        update_freq: int = 20,
        hvp_method: Literal['autograd', 'forward', 'central'] = 'autograd',
        fd_h: float = 1e-3,
        n_samples = 1,
        seed: int | None = None,
        inner: Chainable | None = None
    ):
        defaults = dict(damping=damping, update_freq=update_freq, hvp_method=hvp_method, n_samples=n_samples, fd_h=fd_h, seed=seed)
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
        generator = None
        if seed is not None:
            if 'generator' not in self.global_state:
                self.global_state['generator'] = torch.Generator(params[0].device).manual_seed(seed)
            generator = self.global_state['generator']

        damping = self.get_settings(params, 'damping', cls=NumberList)
        D_sq_acc = self.get_state(params, 'D_sq_acc', cls=TensorList)
        i = self.global_state.get('i', 0)

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        closure = var.closure
        assert closure is not None

        D = None
        if step % update_freq == 0:

            rgrad=None
            for j in range(n_samples):
                u = [torch.randn(p.size(), generator=generator, device=p.device, dtype=p.dtype) for p in params]

                Hvp, rgrad = self.Hvp(u, at_x0=True, var=var, rgrad=rgrad, hvp_method=hvp_method,
                                     h=fd_h, normalize=True, retain_grad=j < n_samples-1)

                if D is None: D = Hvp
                else: torch._foreach_add_(D, Hvp)

            assert D is not None
            if n_samples > 1: torch._foreach_div_(D, n_samples)

            D = TensorList(D)

        update = var.get_update()
        if 'inner' in self.children:
            update = apply_transform(self.children['inner'], tensors=update, params=params, grads=var.grad, var=var)

        var.update, self.global_state['i'] = esgd_(
            tensors_=TensorList(update),
            D=TensorList(D) if D is not None else None,
            D_sq_acc_=D_sq_acc,
            damping=damping,
            update_freq=update_freq,
            step=step,
            i=i,
        )
        return var
