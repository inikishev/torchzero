from typing import Literal
import torch

from ...core import (
    Chainable,
    TensorTransform,
)
from ...utils import NumberList, TensorList, unpack_dicts
from ...linalg.matrix_power import matrix_power as _matrix_power, MatrixPowerMethod

class Adagrad(TensorTransform):
    """Adagrad, divides by sum of past squares of gradients.

    This implementation is identical to ``torch.optim.Adagrad``.

    Args:
        lr_decay (float, optional): learning rate decay. Defaults to 0.
        initial_accumulator_value (float, optional): initial value of the sum of squares of gradients. Defaults to 0.
        eps (float, optional): division epsilon. Defaults to 1e-10.
        alpha (float, optional): step size. Defaults to 1.
        pow (float, optional): power for gradients and accumulator root. Defaults to 2.
        use_sqrt (bool, optional): whether to take the root of the accumulator. Defaults to True.
        inner (Chainable | None, optional): Inner modules that are applied after updating accumulator and before preconditioning. Defaults to None.
    """
    def __init__(
        self,

        # hyperparams
        lr_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        alpha: float = 1,

        # tfms
        inner: Chainable | None = None,
        accumulator_tfm: Chainable | None = None
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner'], defaults["accumulator_tfm"]
        super().__init__(defaults=defaults, inner=inner)

        self.set_child('accumulator', accumulator_tfm)
        self.add_projected_keys("grad", "accumulator")

    @torch.no_grad
    def single_tensor_initialize(self, tensor, param, grad, loss, state, setting):
        state["accumulator"] = torch.full_like(tensor, fill_value=setting["initial_accumulator_value"])

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        torch._foreach_addcmul_([state["accumulator"] for state in states], tensors, tensors)
        self.increment_counter("step", start=0)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        tensors_ = TensorList(tensors)
        step = self.global_state["step"] # 0 on first apply
        eps, alpha, lr_decay = unpack_dicts(settings, "eps", "alpha", "lr_decay", cls=NumberList)

        accumulator = [state["accumulator"] for state in states]
        accumulator = TensorList(self.inner_step_tensors(
            "accumulator", tensors=accumulator, clone=True, params=params, grads=grads, loss=loss, must_exist=False))

        denom = accumulator.sqrt().add_(eps)
        tensors_ /= denom

        clr = alpha / (1 + step * lr_decay)
        tensors_.lazy_mul_(clr)

        return tensors_



class AdagradNorm(TensorTransform):
    """Adagrad-Norm, divides by sum of past means of squares of gradients.

    Args:
        lr_decay (float, optional): learning rate decay. Defaults to 0.
        initial_accumulator_value (float, optional): initial value of the sum of squares of gradients. Defaults to 0.
        eps (float, optional): division epsilon. Defaults to 1e-10.
        alpha (float, optional): step size. Defaults to 1.
        use_sqrt (bool, optional): whether to take the root of the accumulator. Defaults to True.
        inner (Chainable | None, optional): Inner modules that are applied after updating accumulator and before preconditioning. Defaults to None.
    """
    def __init__(
        self,
        lr_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        beta:float | None = None,
        beta_debias: bool = True,
        layerwise: bool = True,
        use_sqrt: bool = True,
        alpha: float = 1,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner']
        super().__init__(defaults=defaults, inner=inner)

    @torch.no_grad
    def multi_tensor_initialize(self, tensors, params, grads, loss, states, settings):

        # layerwise initialize in each state
        if settings[0]["layerwise"]:
            for tensor, state, setting in zip(tensors, states, settings):

                initial_accumulator_value = setting["initial_accumulator_value"]
                state["accumulator"] = torch.tensor(initial_accumulator_value, device=tensor.device, dtype=tensor.dtype)

        # global initialize in global state
        else:
            initial_accumulator_value = settings[0]["initial_accumulator_value"]
            tensor = tensors[0]
            self.global_state["accumulator"] = torch.tensor(initial_accumulator_value, device=tensor.device, dtype=tensor.dtype)

    def _get_accumulator(self, states, settings) -> torch.Tensor | TensorList:
        layerwise = settings[0]["layerwise"]
        if layerwise:
            return TensorList(s["accumulator"] for s in states)

        return self.global_state["accumulator"]

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        accumulator = self._get_accumulator(states, settings)
        self.increment_counter("step", start=0)

        # compute squared gradient norm (gg)
        if isinstance(accumulator, TensorList): gg = tensors.tensorwise_dot(tensors)
        else: gg = tensors.dot(tensors)

        # update the accumulator
        beta = settings[0]["beta"]
        if beta is None: accumulator.add_(gg) # pyright:ignore[reportArgumentType]
        else: accumulator.lerp_(gg, weight=1-beta) # pyright:ignore[reportArgumentType, reportCallIssue]

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        accumulator = self._get_accumulator(states, settings)
        eps, alpha, lr_decay = unpack_dicts(settings, "eps", "alpha", "lr_decay", cls=NumberList)
        step = self.global_state["step"] # 0 on 1st step
        fs = settings[0]
        beta = fs["beta"]

        # ------------------------ debias if beta is not None ------------------------ #
        if fs["beta_debias"] and beta is not None:
            accumulator = accumulator / (1 - beta ** (step + 1))


        # ---------------------------- compute denominator --------------------------- #
        if fs["use_sqrt"]:
            denom = accumulator.sqrt().add_(eps) # pyright:ignore[reportArgumentType]
        else:
            denom = accumulator + eps # pyright:ignore[reportOperatorIssue]


        # ---------------------------- compute the update ---------------------------- #
        tensors /= denom
        clr = alpha / (1 + step * lr_decay) # lr decay
        tensors.lazy_mul_(clr)

        return tensors



class FullMatrixAdagrad(TensorTransform):
    """Full-matrix version of Adagrad, can be customized to make RMSprop or Adam (see examples).

    Note:
        A more memory-efficient version equivalent to full matrix Adagrad on last n gradients is implemented in ``tz.m.GGT``.

    Args:
        reg (float, optional): regularization, scale of identity matrix added to accumulator. Defaults to 1e-12.
        precond_freq (int, optional): frequency of updating the inverse square root of the accumulator. Defaults to 1.
        beta (float | None, optional): momentum for gradient outer product accumulators. if None, uses sum. Defaults to None.
        beta_debias (bool, optional): whether to use debiasing, only has effect when ``beta`` is not ``None``. Defaults to True.
        init (Literal[str], optional):
            how to initialize the accumulator.
            - "identity" - with identity matrix (default).
            - "zeros" - with zero matrix.
            - "ones" - with matrix of ones.
             -"GGT" - with the first outer product
        matrix_power (float, optional): accumulator matrix power. Defaults to -1/2.
        concat_params (bool, optional): if False, each parameter will have it's own accumulator. Defaults to True.
        inner (Chainable | None, optional): inner modules to apply preconditioning to. Defaults to None.

    ## Examples:

    Plain full-matrix adagrad
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.FullMatrixAdagrd(),
        tz.m.LR(1e-2),
    )
    ```

    Full-matrix RMSprop
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.FullMatrixAdagrad(beta=0.99),
        tz.m.LR(1e-2),
    )
    ```

    Full-matrix Adam
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.FullMatrixAdagrad(beta=0.999, inner=tz.m.EMA(0.9)),
        tz.m.Debias(0.9, 0.999),
        tz.m.LR(1e-2),
    )
    ```
    """
    def __init__(
        self,
        reg: float = 1e-12,
        precond_freq: int = 1,
        beta: float | None = None,
        beta_debias: bool=True,
        init: Literal["identity", "zeros", "GGT"] = "identity",
        matrix_power: float = -1/2,
        matrix_power_method: MatrixPowerMethod = "eigh_abs",
        concat_params=True,

        inner: Chainable | None = None,
        accumulator_tfm: Chainable | None = None
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner'], defaults["concat_params"], defaults["accumulator_tfm"]
        super().__init__(defaults=defaults, inner=inner, concat_params=concat_params)

        self.set_child("accumulator", accumulator_tfm)
        self.add_projected_keys("covariance", "accumulator")

    @torch.no_grad
    def single_tensor_update(self, tensor, param, grad, loss, state, setting):

        G = tensor.ravel()
        GGT = torch.outer(G, G)

        # initialize
        if "accumulator" not in state:
            init = setting['init']
            if init == 'identity': state['accumulator'] = torch.eye(GGT.size(0), device=GGT.device, dtype=GGT.dtype)
            elif init == 'zeros': state['accumulator'] =  torch.zeros_like(GGT)
            elif init == 'GGT': state['accumulator'] = GGT.clone()
            else: raise ValueError(init)

        # update
        beta = setting['beta']
        accumulator: torch.Tensor = state["accumulator"]

        if beta is None: accumulator.add_(GGT)
        else: accumulator.lerp_(GGT, 1-beta)

        # update number of GGᵀ in accumulator for divide
        state['num_GGTs'] = state.get('num_GGTs', 0) + 1

    @torch.no_grad
    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        step = state.get('step', 0)
        state['step'] = step + 1

        accumulator: torch.Tensor = state['accumulator']
        accumulator = self.inner_step_tensors("accumulator", [accumulator], clone=True, must_exist=False)[0]

        precond_freq = setting['precond_freq']
        reg = setting['reg']
        beta = setting["beta"]

        # add regularizer
        if reg != 0:
            device = accumulator.device; dtype = accumulator.dtype
            accumulator = accumulator + torch.eye(accumulator.size(0), device=device, dtype=dtype).mul_(reg)

        # for single value use sqrt
        if tensor.numel() == 1:
            dir = tensor.mul_(accumulator.squeeze() ** setting["matrix_power"])

        # otherwise use matrix inverse square root
        else:

            # compute inverse square root and store to state
            try:
                if "B" not in state or step % precond_freq == 0:
                    B = state["B"] = _matrix_power(accumulator, setting["matrix_power"], method=setting["matrix_power_method"])
                else:
                    B = state["B"]

                dir = (B @ tensor.ravel()).view_as(tensor)

            # fallback to diagonal Adagrad on fail
            except torch.linalg.LinAlgError:
                dir = tensor.mul_(accumulator.diagonal() ** setting["matrix_power"])

        # debias
        if setting["beta_debias"] and beta is not None:
            num_GGTs = state.get('num_GGTs', 1)
            bias_correction = 1 - beta ** num_GGTs
            dir *= bias_correction ** 0.5

        return dir
