import torch

from ...core import Chainable, Transform, HVPMethod
from ...utils import Distributions, NumberList, TensorList, unpack_dicts, unpack_states



class SophiaH(Transform):
    """SophiaH optimizer from https://arxiv.org/abs/2305.14342

    This is similar to Adam, but the second momentum is replaced by an exponential moving average of randomized hessian diagonal estimates, and the update is agressively clipped.

    Notes:
        - In most cases SophiaH should be the first module in the chain because it relies on autograd. Use the ``inner`` argument if you wish to apply SophiaH preconditioning to another module's output.

        - This module requires the a closure passed to the optimizer step, as it needs to re-evaluate the loss and gradients for calculating HVPs. The closure must accept a ``backward`` argument (refer to documentation).

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
            Determines how Hessian-vector products are computed.

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
        inner (Chainable | None, optional): preconditioning is applied to the output of this module. Defaults to None.

    ### Examples:

    Using SophiaH:

    ```python

    opt = tz.Optimizer(
        model.parameters(),
        tz.m.SophiaH(),
        tz.m.LR(0.1)
    )
    ```

    SophiaH preconditioner can be applied to any other module by passing it to the ``inner`` argument.
    Turn off SophiaH's first momentum to get just the preconditioning. Here is an example of applying
    SophiaH preconditioning to nesterov momentum (``tz.m.NAG``):

    ```python

    opt = tz.Optimizer(
        model.parameters(),
        tz.m.SophiaH(beta1=0, inner=tz.m.NAG(0.96)),
        tz.m.LR(0.1)
    )
    ```
    """
    def __init__(
        self,
        beta1: float = 0.96,
        beta2: float = 0.99,
        update_freq: int = 10,
        precond_scale: float = 1,
        clip: float = 1,
        eps: float = 1e-12,
        hvp_method: HVPMethod = 'autograd',
        distribution: Distributions = 'gaussian',
        h: float = 1e-3,
        n_samples = 1,
        zHz: bool = True,
        debias: bool = False,
        seed: int | None = None,

        exp_avg_tfm: Chainable | None = None,
        D_exp_avg_tfm: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['exp_avg_tfm'], defaults["D_exp_avg_tfm"]
        super().__init__(defaults)

        self.set_child('exp_avg', exp_avg_tfm)
        self.set_child('D_exp_avg', D_exp_avg_tfm)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        params = objective.params

        beta1, beta2 = unpack_dicts(settings, 'beta1', 'beta2', cls=NumberList)

        exp_avg, D_exp_avg = unpack_states(states, params, 'exp_avg', 'D_exp_avg', cls=TensorList)

        step = self.increment_counter("step", start=0) # 0 on 1st update

        # ---------------------------- hutchinson hessian ---------------------------- #
        fs = settings[0]
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

            D_exp_avg.lerp_(D, weight=1-beta2)

        # --------------------------------- momentum --------------------------------- #
        tensors = objective.get_updates() # do this after hutchinson to not disturb autograd
        exp_avg.lerp_(tensors, 1-beta1)


    @torch.no_grad
    def apply_states(self, objective, states, settings):
        params = objective.params

        beta1, beta2, eps, precond_scale, clip = unpack_dicts(
            settings, 'beta1', 'beta2', 'eps', 'precond_scale', 'clip', cls=NumberList)

        exp_avg, D_exp_avg = unpack_states(states, params, 'exp_avg', 'D_exp_avg')

        # ---------------------------------- debias ---------------------------------- #
        if settings[0]["debias"]:
            bias_correction1 = 1.0 - (beta1 ** (self.global_state["step"] + 1))
            bias_correction2 = 1.0 - (beta2 ** self.global_state["num_Ds"])

            exp_avg = exp_avg / bias_correction1
            D_exp_avg = D_exp_avg / bias_correction2

        # -------------------------------- transforms -------------------------------- #
        exp_avg = TensorList(self.inner_step_tensors(
            "exp_avg", tensors=exp_avg, clone=True, objective=objective, must_exist=False))

        D_exp_avg = TensorList(self.inner_step_tensors(
            "D_exp_avg", tensors=D_exp_avg, clone=True, objective=objective, must_exist=False))

        # ------------------------------ compute update ------------------------------ #
        denom = D_exp_avg.lazy_mul(precond_scale).clip(min=eps)
        objective.updates = (exp_avg / denom).clip_(-clip, clip)
        return objective