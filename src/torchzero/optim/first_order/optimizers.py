from collections.abc import Iterable
from typing import Literal, Unpack

from ...modules import SGD as _SGD
from ...modules import Adagrad as _Adagrad
from ...modules import Adam as _Adam
from ...modules import (
    Centralize,
    LineSearches,
    NesterovMomentum,
    Normalize,
    Sign,
    UseGradSign,
    _CommonKwargs,
    _get_baked_in_and_module_lr,
    _make_common_modules,
)
from ...modules import Lion as _Lion
from ...modules import RMSProp as _RMSProp
from ...modules import Rprop as _Rprop
from ..modular import Modular


class GD(Modular):
    """Gradient descent, by default uses armijo backtracking line search.

    This is technically exactly the same as SGD, but with different defaults.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1).
        line_search (LineSearches | None, optional):
            line search type. Defaults to 'armijo'.
    """
    def __init__(
        self,
        params,
        lr: float = 1,
        line_search: LineSearches | None = 'armijo',
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        kwargs['line_search'] = line_search
        modules = _make_common_modules(None, lr, kwargs)
        super().__init__(params, modules)

class SGD(Modular):
    """Exactly matches `torch.optim.SGD`, except
    nesterov momentum additionally supports dampening, and negative momentum is allowed.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        momentum (float, optional): momentum. Defaults to 0.
        dampening (float, optional): momentum dampening. Defaults to 0.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = _SGD(lr = lr, momentum = momentum, dampening = dampening, weight_decay = weight_decay, nesterov = nesterov)
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)


class SignSGD(Modular):
    """SGD that uses sign of the gradient, can act as a normalizer and improve stability.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        **kwargs: common keyword arguments.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        modules = _make_common_modules(Sign(), lr, kwargs)
        super().__init__(params, modules)


class NormSGD(Modular):
    """SGD with gradient normalization and optionally centralization.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3)
        centralize (bool, optional): whether to centralize gradients (default: True).
        norm_mode (str, optional):
            what to normalize.

            - "global": normalize the entire gradient, as if it was a single vector.

            - "param": normalize each param's gradient.

            - "channel": normalize gradient of each channel of each param (default).
        centralize_mode (str, optional): what to centralize (same options as `norm_mode`). Defaults to 'channel'.
        min_numel (int, optional):
            skips parameters with less than this many elements. This avoids the issue where
            parameters that have a single element always get set to the value of 1.
            Ignored when mode is 'global'. Defaults to 2.
            """
    def __init__(
        self,
        params,
        lr: float = 1e-1,
        centralize=True,
        norm_mode: Literal["global", "param", "channel"] = 'channel',
        centralize_mode: Literal["global", "param", "channel"] = 'channel',
        min_numel=2,
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)

        main: list = [Normalize(lr, mode=norm_mode, min_numel=min_numel)]
        if centralize: main.append(Centralize(centralize_mode, min_numel=min_numel))

        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)


class Adagrad(Modular):
    """Divides ascent direction by mean square root of the sum of all past ascent directions.

    Exactly matches `torch.optim.Adagrad`.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        lr_decay (float, optional): learning rate decay. Defaults to 0.
        initial_accumulator_value (float, optional): initial value of the sum of squares of gradients. Defaults to 0.
        eps (float, optional): term added to the denominator to improve numerical stability. Defaults to 1e-10.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3, lr_decay: float = 0, initial_accumulator_value: float = 0, eps: float = 1e-10,
        **kwargs: Unpack[_CommonKwargs],
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = _Adagrad(lr = lr, lr_decay = lr_decay, initial_accumulator_value = initial_accumulator_value, eps = eps)
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)

class Rprop(Modular):
    """
    Resilient propagation. The update magnitude gets multiplied by `nplus` if gradient didn't change the sign,
    or `nminus` if it did. Then the update is applied with the sign of the current gradient.

    Additionally, if gradient changes sign, the update for that weight is reverted.
    Next step, magnitude for that weight won't change.

    Compared to pytorch this also implements backtracking update when sign changes.
    To make this behave exactly the same as `torch.optim.Rprop`, set `backtrack` to False.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        nplus (float): multiplicative increase factor for when ascent didn't change sign (default: 1.2).
        nminus (float): multiplicative decrease factor for when ascent changed sign (default: 0.5).
        lb (float): minimum step size, can be None (default: 1e-6)
        ub (float): maximum step size, can be None (default: 50)
        backtrack (float):
            if True, when ascent sign changes, undoes last weight update, otherwise sets update to 0.
            When this is False, this exactly matches pytorch Rprop. (default: True)

    reference
        *Riedmiller, M., & Braun, H. (1993, March). A direct adaptive method for faster backpropagation learning:
        The RPROP algorithm. In IEEE international conference on neural networks (pp. 586-591). IEEE.*
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        nplus: float = 1.2,
        nminus: float = 0.5,
        lb: float | None = 1e-6,
        ub: float | None = 50,
        backtrack=True,
        **kwargs: Unpack[_CommonKwargs],
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = _Rprop(lr = lr, nplus = nplus, nminus = nminus, lb=lb, ub = ub, backtrack=backtrack)
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)

class RMSProp(Modular):
    """
    Divides ascent direction by running average of its mean square root.

    Exactly matches `torch.optim.RMSProp`.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        alpha (float, optional):
            smoothing constant (decay of ascent mean square root running average).
            Defaults to 0.99.
        eps (float, optional): term added to the denominator to improve numerical stability. Defaults to 1e-8.
        centered (float, optional):
            if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance.
            Defaults to False.

    reference
        https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    def __init__(
        self,
        params,
        lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8, centered: bool = False,
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        main = _RMSProp(alpha = alpha, eps = eps, centered = centered,)
        modules = _make_common_modules(main, lr, kwargs)
        super().__init__(params, modules)

class Adam(Modular):
    """Adam. Combines momentum and RMSProp. Exactly matches `torch.optim.Adam`.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        beta1 (float, optional): exponential decay rate of gradient moving average. Defaults to 0.9.
        beta2 (float, optional): exponential decay rate of squared gradient moving average. Defaults to 0.999.
        eps (float, optional): epsilon for numerical stability. Defaults to 1e-8.
        amsgrad (bool, optional):
            whether to use the AMSGrad variant of this algorithm from
            On the Convergence of Adam and Beyond (default: False).
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad=False,
        **kwargs: Unpack[_CommonKwargs],
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = _Adam(lr = lr, beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad)
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)

class AdamW(Modular):
    """Adam with decoupled weight decay.
    Weight decay doesn't depend on learning rate and is applied after adam update rule.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        beta1 (float, optional): exponential decay rate of gradient moving average. Defaults to 0.9.
        beta2 (float, optional): exponential decay rate of squared gradient moving average. Defaults to 0.999.
        eps (float, optional): epsilon for numerical stability. Defaults to 1e-8.
        amsgrad (bool, optional):
            whether to use the AMSGrad variant of this algorithm from
            On the Convergence of Adam and Beyond (default: False).
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad=False,
        **kwargs: Unpack[_CommonKwargs],
    ):
        kwargs['decoupled_l2'] = weight_decay
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = _Adam(lr = lr, beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad)
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)


class Grams(Modular):
    """Grams (Gradient Descent with Adaptive Momentum Scaling) from https://arxiv.org/abs/2412.17107v1.
    This is Adam but uses gradient sign.
    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        beta1 (float, optional): exponential decay rate of gradient moving average. Defaults to 0.9.
        beta2 (float, optional): exponential decay rate of squared gradient moving average. Defaults to 0.999.
        eps (float, optional): epsilon for numerical stability. Defaults to 1e-8.
        amsgrad (bool, optional):
            whether to use the AMSGrad variant of this algorithm from
            On the Convergence of Adam and Beyond (default: False).
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad=False,
        **kwargs: Unpack[_CommonKwargs],
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = [
            _Adam(lr = lr, beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad),
            UseGradSign()
        ]
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)


class Lion(Modular):
    """Lion (EvoLved Sign Momentum) optimizer from https://arxiv.org/abs/2302.06675.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        beta1 (float, optional): dampening for momentum. Defaults to 0.9.
        beta2 (float, optional): momentum factor. Defaults to 0.99.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.99,
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        modules = _make_common_modules(_Lion(beta1, beta2), lr, kwargs)
        super().__init__(params, modules)



class NestedNesterov(Modular):
    """Chains multiple nesterov momentums. The default (0.5, 0.5) seems to work well.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        momentums (Iterable[float], optional): sequence of momentums. Defaults to (0.5, 0.5, 0.5).
        dampening (float | Iterable[float], optional):
            sequence of dampenings for each momentum, or a single float that is used
            for all momentums. Defaults to 0.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentums: Iterable[float] = (0.5, 0.5, 0.5),
        dampening: float | Iterable[float] = 0,
        **kwargs: Unpack[_CommonKwargs] # type:ignore
    ):
        momentums = list(momentums)
        if isinstance(dampening, (int, float)): dampening = [dampening for _ in momentums]
        main = [NesterovMomentum(m, d) for m, d in zip(momentums, dampening)]
        modules = _make_common_modules(main, lr_module = lr, kwargs=kwargs)
        super().__init__(params, modules)