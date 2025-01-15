from collections.abc import Iterable
from typing import Literal, Unpack

from ...modules import (
    LR,
    AddNoise,
    Centralize,
    Grad,
    HeavyBall,
    LineSearches, LaplacianSmoothing,
    NesterovMomentum,
    Normalize,
    Random,
    Sign,
    UseGradSign,
    WeightDecay,
    get_line_search,
)
from ...modules import SGD as _SGD
from ...modules import Adagrad as _Adagrad
from ...modules import Adam as _Adam
from ...modules import Lion as _Lion
from ...modules import RMSProp as _RMSProp
from ...modules import Rprop as _Rprop
from ...random.random import Distributions
from ..modular import Modular


class GD(Modular):
    """Gradient descent with armijo line search.

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
    ):
        modules: list = [LR(lr)]
        if line_search is not None: modules.append(get_line_search(line_search))

        super().__init__(params, *modules)

class SGD(Modular):
    """Exactly matches `torch.optim.SGD`, except
    nesterov momentum additionally supports dampening, negative momentum is allowed,
    and weight decay supports decoupling.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        momentum (float, optional): momentum. Defaults to 0.
        dampening (float, optional): momentum dampening. Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
        decoupled=False,
    ):
        modules: list = [
            _SGD(momentum = momentum, dampening = dampening, weight_decay = weight_decay if not decoupled else 0, nesterov = nesterov),
            LR(lr)
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        super().__init__(params, modules)


class SignSGD(Modular):
    """SGD that uses sign of the gradient, can act as a normalizer and improve stability.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        momentum (float, optional): momentum. Defaults to 0.
        dampening (float, optional): momentum dampening. Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
        decoupled=False,
    ):
        modules: list = [
            Sign(),
            _SGD(momentum = momentum, dampening = dampening, weight_decay = weight_decay if not decoupled else 0, nesterov = nesterov),
            LR(lr),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        super().__init__(params, modules)


class NormSGD(Modular):
    """SGD with gradient normalization and optionally centralization.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            learning rate, gradients are normalized to this value.
            This can typically be 10 times bigger than normal SGD (default: 1e-1).
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
        momentum (float, optional): momentum. Defaults to 0.
        dampening (float, optional): momentum dampening. Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.
            """
    def __init__(
        self,
        params,
        lr: float = 1e-1,
        normalize=True,
        norm_mode: Literal["global", "param", "channel"] = 'channel',
        ord = 2,
        centralize=True,
        centralize_mode: Literal["global", "param", "channel"] = 'channel',
        min_numel=2,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list = [
            _SGD(momentum = momentum, dampening = dampening, weight_decay = weight_decay if not decoupled else 0, nesterov = nesterov),
            LR(lr),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        if normalize: modules.insert(0, Normalize(1, mode=norm_mode, min_numel=min_numel, ord=ord))
        if centralize: modules.insert(0, Centralize(centralize_mode, min_numel=min_numel))
        super().__init__(params, modules)


class NoisySGD(Modular):
    """SGD with noise added to gradients. The formula for noise magnitude is `alpha * mean(abs(grad))`.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3)
        alpha (float, optional): magnitude of noise. Defaults to 1e-2.
        distribution (Distributions, optional): distribution of noise. Defaults to 'normal'.
        mode (str, optional):
            how to calculate noise magnitude.

            - "absolute": ignores gradient magnitude and always uses `alpha` as magnitude.

            - "global": multiplies `alpha` by mean of the entire gradient, as if it was a single vector.

            - "param": multiplies `alpha` by mean of each individual parameter (default).

            - "channel": multiplies `alpha` by mean of each channel of each parameter.
        momentum (float, optional): momentum. Defaults to 0.
        dampening (float, optional): momentum dampening. Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        alpha: float = 1,
        distribution: Distributions = 'normal',
        mode: Literal["absolute", "global", "param", "channel"] = "param",
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
        decoupled=False,
    ):

        modules: list = [
            _SGD(momentum = momentum, dampening = dampening, weight_decay = weight_decay if not decoupled else 0, nesterov = nesterov),
            AddNoise(alpha, distribution, mode),
            LR(lr),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        super().__init__(params, modules)

class LaplacianSmoothingSGD(Modular):
    """SGD with laplacian smoothing.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3)
        sigma (float, optional): controls the amount of smoothing. Defaults to 1.
        layerwise (bool, optional):
            If True, applies smoothing to each parameter's gradient separately,
            Otherwise applies it to all gradients, concatenated into a single vector. Defaults to True.
        min_numel (int, optional):
            minimum number of elements in a parameter to apply laplacian smoothing to.
            Only has effect if `layerwise` is True. Defaults to 4.
        momentum (float, optional): momentum. Defaults to 0.
        dampening (float, optional): momentum dampening. Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.

    Reference:
        *Osher, S., Wang, B., Yin, P., Luo, X., Barekat, F., Pham, M., & Lin, A. (2022).
        Laplacian smoothing gradient descent. Research in the Mathematical Sciences, 9(3), 55.*
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        sigma: float = 1,
        layerwise: bool = True,
        min_numel: int = 4,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
        decoupled=False,
    ):

        modules: list = [
            LaplacianSmoothing(sigma=sigma, layerwise=layerwise,min_numel=min_numel),
            _SGD(momentum = momentum, dampening = dampening, weight_decay = 0, nesterov = nesterov),
            LR(lr),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
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
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        lr_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        weight_decay: float = 0,
        decoupled=False,
    ):
        modules: list = [
            _Adagrad(lr_decay = lr_decay, initial_accumulator_value = initial_accumulator_value, eps = eps),
            LR(lr),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
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
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.

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
        weight_decay: float = 0,
        decoupled=False,
    ):
        modules: list = [
            _Rprop(nplus = nplus, nminus = nminus, lb=lb, ub = ub, backtrack=backtrack),
            LR(lr),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
        super().__init__(params, modules)

class RMSProp(Modular):
    """
    Divides ascent direction by running average of its mean square root.

    Exactly matches `torch.optim.RMSProp`, except momentum initialization is arbitrarily different.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        momentum (float, optional): momentum. Defaults to 0.
        alpha (float, optional):
            smoothing constant (decay of ascent mean square root running average).
            Defaults to 0.99.
        eps (float, optional): term added to the denominator to improve numerical stability. Defaults to 1e-8.
        centered (float, optional):
            if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance.
            Defaults to False.
        dampening (float, optional): momentum dampening. Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.

    reference
        https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0,
        alpha: float = 0.99,
        eps: float = 1e-8,
        centered: bool = False,
        nesterov = False,
        dampening: float = 0,
        weight_decay: float = 0,
        decoupled=False,
    ):
        modules: list = [
            _RMSProp(smoothing = alpha, eps = eps, centered = centered,),
            _SGD(momentum = momentum, dampening = dampening, weight_decay = 0, nesterov = nesterov),
            LR(lr),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
        super().__init__(params, modules)

class Adam(Modular):
    """Adam. Combines momentum and RMSProp. Exactly matches `torch.optim.Adam`, except
    if `decoupled` is True, weight decay is truly decoupled and doesn't depend on LR.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        beta1 (float, optional): exponential decay rate of gradient moving average. Defaults to 0.9.
        beta2 (float, optional): exponential decay rate of squared gradient moving average. Defaults to 0.999.
        eps (float, optional): epsilon for numerical stability. Defaults to 1e-8.
        amsgrad (bool, optional):
            whether to use the AMSGrad variant of this algorithm from
            On the Convergence of Adam and Beyond (default: False).
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad=False,
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list = [
            _Adam(beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad),
            LR(lr),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
        super().__init__(params, modules)

class AdamW(Adam):
    """AdamW. Combines momentum and RMSProp. Exactly matches `torch.optim.Adam`, except
    if `decoupled` is True, weight decay is truly decoupled and doesn't depend on LR.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        beta1 (float, optional): exponential decay rate of gradient moving average. Defaults to 0.9.
        beta2 (float, optional): exponential decay rate of squared gradient moving average. Defaults to 0.999.
        eps (float, optional): epsilon for numerical stability. Defaults to 1e-8.
        amsgrad (bool, optional):
            whether to use the AMSGrad variant of this algorithm from
            On the Convergence of Adam and Beyond (default: False).
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.01.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad=False,
        weight_decay: float = 1e-2,
        decoupled=True,
    ):
        super().__init__(params=params,lr=lr,beta1=beta1,beta2=beta2,eps=eps,amsgrad=amsgrad,weight_decay=weight_decay,decoupled=decoupled)

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
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad=False,
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list = [
            _Adam(beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad),
            LR(lr),
            UseGradSign()
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
        super().__init__(params, modules)


class Lion(Modular):
    """Lion (EvoLved Sign Momentum) optimizer from https://arxiv.org/abs/2302.06675.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        beta1 (float, optional): dampening for momentum. Defaults to 0.9.
        beta2 (float, optional): momentum factor. Defaults to 0.99.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list = [
            _Lion(beta1, beta2),
            LR(lr)
            ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
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
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to True.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.

    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentums: Iterable[float] = (0.5, 0.5, 0.5),
        dampening: float | Iterable[float] = 0,
        nesterov=True,
        weight_decay: float = 0,
        decoupled=True,
    ):
        momentums = list(momentums)
        if isinstance(dampening, (int, float)): dampening = [dampening for _ in momentums]

        cls = NesterovMomentum if nesterov else HeavyBall
        modules: list = [cls(m, d) for m, d in zip(momentums, dampening)] + [LR(lr)]

        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))

        super().__init__(params, modules)