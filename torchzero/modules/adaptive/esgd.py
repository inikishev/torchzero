from typing import Literal

import torch

from ...core import Chainable, Module, step, HVPMethod
from ...utils import Distributions, NumberList, TensorList


def esgd_(
    tensors_: TensorList,
    Hz: TensorList | None,
    Hz_sq_acc_: TensorList,
    damping: float | NumberList,
    update_freq: int,
    step: int,
    i: int,
):
    # update preconditioner
    if step % update_freq == 0:
        assert Hz is not None
        Hz_sq_acc_.addcmul_(Hz, Hz)
        i += 1
    else:
        assert Hz is None

    denom = (Hz_sq_acc_ / max(i, 1)).sqrt_().add_(damping)
    return tensors_.div_(denom), i


class ESGD(Module):
    """Equilibrated Gradient Descent (https://arxiv.org/abs/1502.04390)

    This is similar to Adagrad, but the accumulates squared randomized hessian diagonal estimates instead of squared gradients.

    Notes:
        - In most cases ESGD should be the first module in the chain because it relies on autograd. Use the ``inner`` argument if you wish to apply ESGD preconditioning to another module's output.

        - This module requires a closure passed to the optimizer step, as it needs to re-evaluate the loss and gradients for calculating HVPs. The closure must accept a ``backward`` argument (refer to documentation).

    Args:
        damping (float, optional): added to denominator for stability. Defaults to 1e-4.
        update_freq (int, optional):
            frequency of updating hessian diagonal estimate via a hessian-vector product.
            This value can be increased to reduce computational cost. Defaults to 20.
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
            2. pass inputs to :code:`inner`.
            3. momentum and preconditioning are applied to the ouputs of :code:`inner`.

    ### Examples:

    Using ESGD:
```python

    opt = tz.Modular(
        model.parameters(),
        tz.m.ESGD(),
        tz.m.LR(0.1)
    )
    ```

    ESGD preconditioner can be applied to any other module by passing it to the :code:`inner` argument. Here is an example of applying
    ESGD preconditioning to nesterov momentum (:code:`tz.m.NAG`):

    ```python
    opt = tz.Modular(
        model.parameters(),
        tz.m.ESGD(beta1=0, inner=tz.m.NAG(0.9)),
        tz.m.LR(0.1)
    )
    ```

    """
    def __init__(
        self,
        damping: float = 1e-4,
        update_freq: int = 20,
        distribution: Distributions = 'gaussian',
        hvp_method: HVPMethod = 'autograd',
        h: float = 1e-3,
        n_samples = 1,
        zHz: bool = False,
        seed: int | None = None,
        inner: Chainable | None = None
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner']
        super().__init__(defaults)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def apply(self, var):
        params = var.params

        damping = self.get_settings(params, 'damping', cls=NumberList)
        Hz_sq_acc = self.get_state(params, 'Hz_sq_acc', cls=TensorList)
        i = self.global_state.get('i', 0)

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        closure = var.closure
        assert closure is not None

        Hz = None
        update_freq = self.defaults['update_freq']
        if step % update_freq == 0:

            Hz, _ = var.hutchinson_hessian(
                rgrad = None,
                at_x0 = True,
                n_samples = self.defaults['n_samples'],
                distribution = self.defaults['distribution'],
                hvp_method = self.defaults['hvp_method'],
                h = self.defaults['h'],
                zHz = self.defaults["zHz"], # False
                generator = self.get_generator(params[0].device, self.defaults["seed"]),
            )

            Hz = TensorList(Hz)

        update = var.get_update()
        if 'inner' in self.children:
            update = step(self.children['inner'], tensors=update, params=params, grads=var.grad, var=var)

        var.update, self.global_state['i'] = esgd_(
            tensors_=TensorList(update),
            Hz=TensorList(Hz) if Hz is not None else None,
            Hz_sq_acc_=Hz_sq_acc,
            damping=damping,
            update_freq=update_freq,
            step=step,
            i=i,
        )

        return var
