from collections.abc import Callable
from typing import Literal

import torch

from ...core import Chainable, Module, Target, Transform, apply_transform
from ...utils import Distributions, NumberList, TensorList, as_tensorlist


def sophia_H(
    tensors: TensorList,
    D: TensorList | None,
    exp_avg_: TensorList,
    D_exp_avg_: TensorList,
    beta1: float | NumberList,
    beta2: float | NumberList,
    update_freq: int,
    precond_scale: float | NumberList,
    clip: float | NumberList,
    eps: float | NumberList,
    step: int
):
    # momentum
    exp_avg_.lerp_(tensors, 1-beta1)

    # update preconditioner
    if step % update_freq == 0:
        assert D is not None
        D_exp_avg_.lerp_(D, 1-beta2)

    else:
        assert D is None

    denom = (D_exp_avg_ * precond_scale).clip_(min=eps)
    return (exp_avg_ / denom).clip_(-clip, clip)


class SophiaH(Module):
    """SophiaH optimizer from https://arxiv.org/abs/2305.14342

    This is similar to Adam, but the second momentum is replaced by an exponential moving average of randomized hessian diagonal estimates, and the update is agressively clipped.

    .. note::
        In most cases SophiaH should be the first module in the chain because it relies on autograd. Use the :code:`inner` argument if you wish to apply SophiaH preconditioning to another module's output.

    .. note::
        If you are using gradient estimators or reformulations, set :code:`hvp_method` to "forward" or "central".

    .. note::
        This module requires the a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients for calculating HVPs.
        The closure must accept a ``backward`` argument (refer to documentation).

    Args:
        beta1 (float, optional): first momentum. Defaults to 0.96.
        beta2 (float, optional): momentum for hessian diagonal estimate. Defaults to 0.99.
        update_freq (int, optional):
            frequency of updating hessian diagonal estimate via a hessian-vector product. Defaults to 10.
        precond_scale (float, optional):
            scale of the preconditioner. Defaults to 1.
        clip (float, optional):
            clips update to (-clip, clip). Defaults to 1.
        eps (float, optional):
            clips hessian diagonal esimate to be no less than this value. Defaults to 1e-12.
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
        fd_h (float, optional): finite difference step size if :code:`hvp_method` is "forward" or "central". Defaults to 1e-3.
        n_samples (int, optional):
            number of hessian-vector products with random vectors to evaluate each time when updating
            the preconditioner. Larger values may lead to better hessian diagonal estimate. Defaults to 1.
        seed (int | None, optional): seed for random vectors. Defaults to None.
        inner (Chainable | None, optional): preconditioning is applied to the output of this module. Defaults to None.

    Examples:
        Using SophiaH:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.SophiaH(),
                tz.m.LR(0.1)
            )

        SophiaH preconditioner can be applied to any other module by passing it to the :code:`inner` argument.
        Turn off SophiaH's first momentum to get just the preconditioning. Here is an example of applying
        SophiaH preconditioning to nesterov momentum (:code:`tz.m.NAG`):

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.SophiaH(beta1=0, inner=tz.m.NAG(0.96)),
                tz.m.LR(0.1)
            )

    """
    def __init__(
        self,
        beta1: float = 0.96,
        beta2: float = 0.99,
        update_freq: int = 10,
        precond_scale: float = 1,
        clip: float = 1,
        eps: float = 1e-12,
        hvp_method: Literal['batched', 'autograd', 'forward', 'central'] = 'autograd',
        distribution: Distributions = 'gaussian',
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

        beta1, beta2, precond_scale, clip, eps = self.get_settings(params,
            'beta1', 'beta2', 'precond_scale', 'clip', 'eps', cls=NumberList)

        exp_avg, D_exp_avg = self.get_state(params, 'exp_avg', 'D_exp_avg', cls=TensorList)

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


        update = var.get_update()
        if 'inner' in self.children:
            update = apply_transform(self.children['inner'], tensors=update, params=params, grads=var.grad, var=var)

        var.update = sophia_H(
            tensors=TensorList(update),
            D=TensorList(D) if D is not None else None,
            exp_avg_=exp_avg,
            D_exp_avg_=D_exp_avg,
            beta1=beta1,
            beta2=beta2,
            update_freq=update_freq,
            precond_scale=precond_scale,
            clip=clip,
            eps=eps,
            step=step,
        )
        return var
