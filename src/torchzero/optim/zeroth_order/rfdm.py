from typing import Literal, Unpack

import torch

from ...modules import SGD, Wrap
from ...modules import RandomizedFDM as _RandomizedFDM
from ...modules import WeightDecay
from ...modules.gradient_approximation._fd_formulas import _FD_Formulas
from ...tensorlist import Distributions
from ..modular import Modular


class RandomizedFDM(Modular):
    """Randomized finite difference gradient approximation (e.g. SPSA, RDSA, Nesterov random search).

    With `forward` and `backward` formulas performs `1 + n_samples` evaluations per step;
    with `central` formula performs `2 * n_samples` evaluations per step.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. Defaults to 1e-3.
        eps (float, optional): finite difference epsilon. Defaults to 1e-3.
        formula (_FD_Formulas, optional): finite difference formula. Defaults to "forward".
        n_samples (int, optional): number of random gradient approximations that will be averaged. Defaults to 1.
        distribution (Distributions, optional): distribution for random perturbations. Defaults to "normal".
        randomize_every (int, optional): number of steps between randomizing perturbations. Defaults to 1.
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
        eps: float = 1e-3,
        formula: _FD_Formulas = "forward",
        n_samples: int = 1,
        distribution: Distributions = "normal",
        randomize_every: int = 1,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
        decoupled=False,
    ):
        modules: list = [
            _RandomizedFDM(
                eps=eps,
                formula=formula,
                n_samples=n_samples,
                distribution=distribution,
                randomize_every=randomize_every,
            ),
            SGD(lr = lr, momentum = momentum, dampening = dampening, weight_decay = weight_decay if not decoupled else 0, nesterov = nesterov)
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        super().__init__(params, modules)


class SPSA(RandomizedFDM):
    """Simultaneous perturbation stochastic approximation method.
    This is the same as a randomized finite difference method with central formula
    and perturbations taken from rademacher distibution.
    Due to rademacher having values -1 or 1, the original formula divides by the perturbation,
    but that is equivalent to multiplying by it, which is the same as central difference formula.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. Defaults to 1e-3.
        eps (float, optional): finite difference epsilon. Defaults to 1e-3.
        momentum (float, optional): momentum factor. Defaults to 0.
        weight_decay (float, optional): weight decay (L2 penalty). Defaults to 0.
        dampening (float, optional): dampening for momentum. Defaults to 0.
        nesterov (bool, optional): enables Nesterov momentum (supports dampening). Defaults to False.
        formula (_FD_Formulas, optional): finite difference formula. Defaults to "central".
        n_samples (int, optional): number of random gradient approximations that will be averaged. Defaults to 1.
        distribution (Distributions, optional): distribution for random perturbations. Defaults to "rademacher".
        randomize_every (int, optional): number of steps between randomizing perturbations. Defaults to 1.
        momentum (float, optional): momentum. Defaults to 0.
        dampening (float, optional): momentum dampening. Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.

    reference
        *Spall, J. C. (1992), “Multivariate Stochastic Approximation Using a Simultaneous Perturbation
        Gradient Approximation,” IEEE Transactions on Automatic Control, vol. 37(3), pp. 332–341.*
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        eps: float = 1e-3,
        formula: _FD_Formulas = "central",
        n_samples: int = 1,
        distribution: Distributions = 'rademacher',
        randomize_every: int = 1,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
        decoupled=False,    ):
        super().__init__(
            params = params,
            lr = lr,
            eps = eps,
            formula = formula,
            n_samples = n_samples,
            distribution = distribution,
            randomize_every=randomize_every,
            momentum = momentum,
            dampening = dampening,
            nesterov = nesterov,
            weight_decay = weight_decay,
            decoupled = decoupled,
        )


class RandomGaussianSmoothing(RandomizedFDM):
    """Random search with gaussian smoothing.
    This is similar to forward randomized finite difference method, and it
    approximates and averages the gradient with multiple random perturbations taken from normal distribution,
    which is an approximation for the gradient of a gaussian smoothed version of the objective function.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. Defaults to 1e-2.
        eps (float, optional): finite difference epsilon. Defaults to 1e-2.
        momentum (float, optional): momentum factor. Defaults to 0.
        weight_decay (float, optional): weight decay (L2 penalty). Defaults to 0.
        dampening (float, optional): dampening for momentum. Defaults to 0.
        nesterov (bool, optional): enables Nesterov momentum (supports dampening). Defaults to False.
        formula (_FD_Formulas, optional): finite difference formula. Defaults to "forward".
        n_samples (int, optional): number of random gradient approximations that will be averaged. Defaults to 1.
        distribution (Distributions, optional): distribution for random perturbations. Defaults to "normal".
        randomize_every (int, optional): number of steps between randomizing perturbations. Defaults to 1.
        momentum (float, optional): momentum. Defaults to 0.
        dampening (float, optional): momentum dampening. Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.

    reference
        *Nesterov, Y., & Spokoiny, V. (2017).
        Random gradient-free minimization of convex functions.
        Foundations of Computational Mathematics, 17(2), 527-566.*

    """
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        eps: float = 1e-2,
        formula: _FD_Formulas = "forward",
        n_samples: int = 10,
        distribution: Distributions = 'normal',
        randomize_every: int = 1,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
        decoupled=False
    ):
        super().__init__(
            params = params,
            lr = lr,
            eps = eps,
            formula = formula,
            n_samples = n_samples,
            distribution = distribution,
            randomize_every=randomize_every,
            momentum = momentum,
            dampening = dampening,
            nesterov = nesterov,
            weight_decay = weight_decay,
            decoupled = decoupled,
        )

class RandomizedFDMWrapper(Modular):
    """Randomized finite difference gradient approximation (e.g. SPSA, RDSA, Nesterov random search).

    With `forward` and `backward` formulas performs `1 + n_samples` evaluations per step;
    with `central` formula performs `2 * n_samples` evaluations per step.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        optimizer (torch.optim.Optimizer): optimizer that will perform optimization using RFDM-approximated gradients.
        eps (float, optional): finite difference epsilon. Defaults to 1e-3.
        formula (_FD_Formulas, optional): finite difference formula. Defaults to "forward".
        n_samples (int, optional): number of random gradient approximations that will be averaged. Defaults to 1.
        distribution (Distributions, optional): distribution for random perturbations. Defaults to "normal".
        randomize_every (int, optional): number of steps between randomizing perturbations. Defaults to 1.
        randomize_closure (bool, optional): whether to generate a new random perturbation each time closure
            is evaluated with `backward=True` (this ignores `randomize_every`). Defaults to False. Defaults to False.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        eps: float = 1e-3,
        formula: _FD_Formulas = "forward",
        n_samples: int = 1,
        distribution: Distributions = "normal",
        randomize_every: int = 1,
        randomize_closure: bool = False,
    ):
        modules = [
            _RandomizedFDM(
                eps=eps,
                formula=formula,
                n_samples=n_samples,
                distribution=distribution,
                randomize_every=randomize_every,
                randomize_closure = randomize_closure,
                make_closure=True,
            ),
            Wrap(optimizer, pass_closure=True)
        ]
        super().__init__(optimizer.param_groups, modules)