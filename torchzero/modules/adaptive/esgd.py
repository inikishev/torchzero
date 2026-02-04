from typing import Literal

import torch

from ...core import Chainable, HVPMethod, Transform
from ...utils import Distributions, NumberList, TensorList, unpack_dicts, unpack_states


class ESGD(Transform):
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

    opt = tz.Optimizer(
        model.parameters(),
        tz.m.ESGD(),
        tz.m.LR(0.1)
    )
    ```

    ESGD preconditioner can be applied to any other module by passing it to the :code:`inner` argument. Here is an example of applying
    ESGD preconditioning to nesterov momentum (:code:`tz.m.NAG`):

    ```python
    opt = tz.Optimizer(
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
        beta: float | None = None,
        beta_debias: bool = True,

        inner: Chainable | None = None,
        Hz_sq_acc_tfm: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner'], defaults["Hz_sq_acc_tfm"]
        super().__init__(defaults, inner=inner)

        self.set_child("Hz_sq_acc", Hz_sq_acc_tfm)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        params = objective.params

        fs = settings[0]
        update_freq = fs['update_freq']

        # ------------------------------- accumulate Hz ------------------------------ #
        step = self.increment_counter("step", start=0)

        if step % update_freq == 0:
            self.increment_counter("num_Hzs", start=1)

            Hz, _ = objective.hutchinson_hessian(
                rgrad = None,
                at_x0 = True,
                n_samples = fs['n_samples'],
                distribution = fs['distribution'],
                hvp_method = fs['hvp_method'],
                h = fs['h'],
                zHz = fs["zHz"], # default is False, so it returns Hz, not z⊙Hz
                generator = self.get_generator(params[0].device, fs["seed"]),
            )

            Hz = TensorList(Hz)
            Hz_sq_acc = unpack_states(states, params, 'Hz_sq_acc', cls=TensorList)

            beta = fs["beta"]
            if beta is None:
                Hz_sq_acc.addcmul_(Hz, Hz)

            else:
                Hz_sq_acc.mul_(beta).addcmul_(Hz, Hz, value=1-beta)

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        tensors = TensorList(objective.get_updates())
        Hz_sq_acc = unpack_states(states, tensors, 'Hz_sq_acc', cls=TensorList)
        num_Hzs = self.global_state["num_Hzs"]
        fs = settings[0]

        # ---------------------------------- debias ---------------------------------- #
        beta = fs["beta"]
        beta_debias = fs["beta_debias"]

        if beta_debias and beta is not None:
            bias_correction = 1.0 - beta ** num_Hzs
            Hz_sq_acc = Hz_sq_acc / bias_correction

        else:
            Hz_sq_acc = Hz_sq_acc / num_Hzs

        # ---------------------------------- update ---------------------------------- #
        damping = [s["damping"] for s in settings]

        denom = (Hz_sq_acc / num_Hzs).sqrt_().add_(damping)

        objective.updates = tensors.div_(denom)
        return objective