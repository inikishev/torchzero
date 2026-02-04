from collections.abc import Callable
from typing import Any
from functools import partial
import torch

from ...utils import TensorList, Distributions, NumberList
from .grad_approximator import GradApproximator, GradTarget, _FD_Formula

def _rforward2(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    """p_fn is a function that returns the perturbation.
    It may return pre-generated one or generate one deterministically from a seed as in MeZO.
    Returned perturbation must be multiplied by `h`."""
    if f_0 is None: f_0 = closure(False)
    params += p_fn()
    f_1 = closure(False)
    params -= p_fn()
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_0, (f_1 - f_0) / h # (loss, loss_approx, grad)

def _rbackward2(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    if f_0 is None: f_0 = closure(False)
    params -= p_fn()
    f_m1 = closure(False)
    params += p_fn()
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_0, (f_0 - f_m1) / h

def _rcentral2(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: Any):
    params += p_fn()
    f_1 = closure(False)

    params -= p_fn() * 2
    f_m1 = closure(False)

    params += p_fn()
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_1, (f_1 - f_m1) / (2 * h)

def _rforward3(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    if f_0 is None: f_0 = closure(False)
    params += p_fn()
    f_1 = closure(False)

    params += p_fn()
    f_2 = closure(False)

    params -= p_fn() * 2
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_0, (-3*f_0 + 4*f_1 - f_2) / (2 * h)

def _rbackward3(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    if f_0 is None: f_0 = closure(False)

    params -= p_fn()
    f_m1 = closure(False)

    params -= p_fn()
    f_m2 = closure(False)

    params += p_fn() * 2
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_0, (f_m2 - 4*f_m1 + 3*f_0) / (2 * h)

def _rcentral4(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    params += p_fn()
    f_1 = closure(False)

    params += p_fn()
    f_2 = closure(False)

    params -= p_fn() * 3
    f_m1 = closure(False)

    params -= p_fn()
    f_m2 = closure(False)

    params += p_fn() * 2
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_1, (f_m2 - 8*f_m1 + 8*f_1 - f_2) / (12 * h)

# some good ones
# Pachalyl S. et al. Generalized simultaneous perturbation-based gradient search with reduced estimator bias //IEEE Transactions on Automatic Control. – 2025.
# Three measurements GSPSA is _rforward3
# Four measurements GSPSA
def _rforward4(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    if f_0 is None: f_0 = closure(False)
    params += p_fn()
    f_1 = closure(False)

    params += p_fn()
    f_2 = closure(False)

    params += p_fn()
    f_3 = closure(False)

    params -= p_fn() * 3
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_0, (2*f_3 - 9*f_2 + 18*f_1 - 11*f_0) / (6 * h)

def _rforward5(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    if f_0 is None: f_0 = closure(False)
    params += p_fn()
    f_1 = closure(False)

    params += p_fn()
    f_2 = closure(False)

    params += p_fn()
    f_3 = closure(False)

    params += p_fn()
    f_4 = closure(False)

    params -= p_fn() * 4
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_0, (-3*f_4 + 16*f_3 - 36*f_2 + 48*f_1 - 25*f_0) / (12 * h)

# # another central4
# def _bgspsa4(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
#     params += p_fn()
#     f_1 = closure(False)

#     params += p_fn() * 2
#     f_3 = closure(False)

#     params -= p_fn() * 4
#     f_m1 = closure(False)

#     params -= p_fn() * 2
#     f_m3 = closure(False)

#     params += p_fn() * 3
#     h = h**2 # because perturbation already multiplied by h
#     return f_0, f_1, (27*f_1 - f_m1 - f_3 + f_m3) / (48 * h)


_RFD_FUNCS: dict[_FD_Formula, Callable] = {
    "forward": _rforward2,
    "forward2": _rforward2,
    "backward": _rbackward2,
    "backward2": _rbackward2,
    "central": _rcentral2,
    "central2": _rcentral2,
    "central3": _rcentral2,
    "forward3": _rforward3,
    "backward3": _rbackward3,
    "central4": _rcentral4,
    "forward4": _rforward4,
    "forward5": _rforward5,
    # "bspsa4": _bgspsa4,
}


class RandomizedFDM(GradApproximator):
    """Gradient approximation via a randomized finite-difference method.

    Note:
        This module is a gradient approximator. It modifies the closure to evaluate the estimated gradients,
        and further closure-based modules will use the modified closure. All modules after this will use estimated gradients.

    Args:
        h (float, optional): finite difference step size of jvp_method is set to `forward` or `central`. Defaults to 1e-3.
        n_samples (int, optional): number of random gradient samples. Defaults to 1.
        formula (_FD_Formula, optional): finite difference formula. Defaults to 'central2'.
        distribution (Distributions, optional): distribution. Defaults to "rademacher".
            If this is set to a value higher than zero, instead of using directional derivatives in a new random direction on each step, the direction changes gradually with momentum based on this value. This may make it possible to use methods with memory. Defaults to 0.
        pre_generate (bool, optional):
            whether to pre-generate gradient samples before each step. If samples are not pre-generated, whenever a method performs multiple closure evaluations, the gradient will be evaluated in different directions each time. Defaults to True.
        seed (int | None | torch.Generator, optional): Seed for random generator. Defaults to None.
        target (GradTarget, optional): what to set on var. Defaults to "closure".

    Examples:
    #### Simultaneous perturbation stochastic approximation (SPSA) method

    SPSA is randomized FDM with rademacher distribution and central formula.
    ```py
    spsa = tz.Optimizer(
        model.parameters(),
        tz.m.RandomizedFDM(formula="fd_central", distribution="rademacher"),
        tz.m.LR(1e-2)
    )
    ```

    #### Random-direction stochastic approximation (RDSA) method

    RDSA is randomized FDM with usually gaussian distribution and central formula.
    ```
    rdsa = tz.Optimizer(
        model.parameters(),
        tz.m.RandomizedFDM(formula="fd_central", distribution="gaussian"),
        tz.m.LR(1e-2)
    )
    ```

    #### Gaussian smoothing method

    GS uses many gaussian samples with possibly a larger finite difference step size.
    ```
    gs = tz.Optimizer(
        model.parameters(),
        tz.m.RandomizedFDM(n_samples=100, distribution="gaussian", formula="forward2", h=1e-1),
        tz.m.NewtonCG(hvp_method="forward"),
        tz.m.Backtracking()
    )
    ```

    #### RandomizedFDM with momentum

    Momentum might help by reducing the variance of the estimated gradients.
    ```
    momentum_spsa = tz.Optimizer(
        model.parameters(),
        tz.m.RandomizedFDM(),
        tz.m.HeavyBall(0.9),
        tz.m.LR(1e-3)
    )
    ```
    """
    PRE_MULTIPLY_BY_H = True
    def __init__(
        self,
        h: float = 1e-3,
        n_samples: int = 1,
        formula: _FD_Formula = "central",
        distribution: Distributions = "rademacher",
        pre_generate: bool = True,
        return_approx_loss: bool = False,
        seed: int | None | torch.Generator = None,
        target: GradTarget = "closure",
    ):
        defaults = dict(h=h, formula=formula, n_samples=n_samples, distribution=distribution, pre_generate=pre_generate, seed=seed)
        super().__init__(defaults, return_approx_loss=return_approx_loss, target=target)


    def pre_step(self, objective):
        h = self.get_settings(objective.params, 'h')
        pre_generate = self.defaults['pre_generate']

        if pre_generate:
            n_samples = self.defaults['n_samples']
            distribution = self.defaults['distribution']

            params = TensorList(objective.params)
            generator = self.get_generator(params[0].device, self.defaults['seed'])
            perturbations = [params.sample_like(distribution=distribution, variance=1, generator=generator) for _ in range(n_samples)]

            # this is false for ForwardGradient where h isn't used and it subclasses this
            if self.PRE_MULTIPLY_BY_H:
                torch._foreach_mul_([p for l in perturbations for p in l], [v for vv in h for v in [vv]*n_samples])

            for param, prt in zip(params, zip(*perturbations)):
                self.state[param]['perturbations'] = prt

    @torch.no_grad
    def approximate(self, closure, params, loss):
        params = TensorList(params)
        loss_approx = None

        h = NumberList(self.settings[p]['h'] for p in params)
        n_samples = self.defaults['n_samples']
        distribution = self.defaults['distribution']
        fd_fn = _RFD_FUNCS[self.defaults['formula']]

        default = [None]*n_samples
        perturbations = list(zip(*(self.state[p].get('perturbations', default) for p in params)))
        generator = self.get_generator(params[0].device, self.defaults['seed'])

        grad = None
        for i in range(n_samples):
            prt = perturbations[i]

            if prt[0] is None:
                prt = params.sample_like(distribution=distribution, generator=generator, variance=1).mul_(h)

            else: prt = TensorList(prt)

            loss, loss_approx, d = fd_fn(closure=closure, params=params, p_fn=lambda: prt, h=h, f_0=loss)
            # here `d` is a numberlist of directional derivatives, due to per parameter `h` values.

            # support for per-sample values which gives better estimate
            if d[0].numel() > 1: d = d.map(torch.mean)

            if grad is None: grad = prt * d
            else: grad += prt * d

        assert grad is not None
        if n_samples > 1: grad.div_(n_samples)

        # mean if got per-sample values
        if loss is not None:
            if loss.numel() > 1:
                loss = loss.mean()

        if loss_approx is not None:
            if loss_approx.numel() > 1:
                loss_approx = loss_approx.mean()

        return grad, loss, loss_approx

class SPSA(RandomizedFDM):
    """
    Gradient approximation via Simultaneous perturbation stochastic approximation (SPSA) method.

    Note:
        This module is a gradient approximator. It modifies the closure to evaluate the estimated gradients,
        and further closure-based modules will use the modified closure. All modules after this will use estimated gradients.

    Args:
        h (float, optional): finite difference step size of jvp_method is set to `forward` or `central`. Defaults to 1e-3.
        n_samples (int, optional): number of random gradient samples. Defaults to 1.
        formula (_FD_Formula, optional): finite difference formula. Defaults to 'central2'.
        distribution (Distributions, optional): distribution. Defaults to "rademacher".
        pre_generate (bool, optional):
            whether to pre-generate gradient samples before each step. If samples are not pre-generated, whenever a method performs multiple closure evaluations, the gradient will be evaluated in different directions each time. Defaults to True.
        seed (int | None | torch.Generator, optional): Seed for random generator. Defaults to None.
        target (GradTarget, optional): what to set on var. Defaults to "closure".

    References:
        Chen, Y. (2021). Theoretical study and comparison of SPSA and RDSA algorithms. arXiv preprint arXiv:2107.12771. https://arxiv.org/abs/2107.12771
    """

class RDSA(RandomizedFDM):
    """
    Gradient approximation via Random-direction stochastic approximation (RDSA) method.

    Note:
        This module is a gradient approximator. It modifies the closure to evaluate the estimated gradients,
        and further closure-based modules will use the modified closure. All modules after this will use estimated gradients.

    Args:
        h (float, optional): finite difference step size of jvp_method is set to `forward` or `central`. Defaults to 1e-3.
        n_samples (int, optional): number of random gradient samples. Defaults to 1.
        formula (_FD_Formula, optional): finite difference formula. Defaults to 'central2'.
        distribution (Distributions, optional): distribution. Defaults to "gaussian".
        pre_generate (bool, optional):
            whether to pre-generate gradient samples before each step. If samples are not pre-generated, whenever a method performs multiple closure evaluations, the gradient will be evaluated in different directions each time. Defaults to True.
        seed (int | None | torch.Generator, optional): Seed for random generator. Defaults to None.
        target (GradTarget, optional): what to set on var. Defaults to "closure".

    References:
        Chen, Y. (2021). Theoretical study and comparison of SPSA and RDSA algorithms. arXiv preprint arXiv:2107.12771. https://arxiv.org/abs/2107.12771

    """
    def __init__(
        self,
        h: float = 1e-3,
        n_samples: int = 1,
        formula: _FD_Formula = "central2",
        distribution: Distributions = "gaussian",
        pre_generate: bool = True,
        return_approx_loss: bool = False,
        target: GradTarget = "closure",
        seed: int | None | torch.Generator = None,
    ):
        super().__init__(h=h, n_samples=n_samples,formula=formula,distribution=distribution,pre_generate=pre_generate,target=target,seed=seed, return_approx_loss=return_approx_loss)

class GaussianSmoothing(RandomizedFDM):
    """
    Gradient approximation via Gaussian smoothing method.

    Note:
        This module is a gradient approximator. It modifies the closure to evaluate the estimated gradients,
        and further closure-based modules will use the modified closure. All modules after this will use estimated gradients.

    Args:
        h (float, optional): finite difference step size of jvp_method is set to `forward` or `central`. Defaults to 1e-2.
        n_samples (int, optional): number of random gradient samples. Defaults to 100.
        formula (_FD_Formula, optional): finite difference formula. Defaults to 'forward2'.
        distribution (Distributions, optional): distribution. Defaults to "gaussian".
        pre_generate (bool, optional):
            whether to pre-generate gradient samples before each step. If samples are not pre-generated, whenever a method performs multiple closure evaluations, the gradient will be evaluated in different directions each time. Defaults to True.
        seed (int | None | torch.Generator, optional): Seed for random generator. Defaults to None.
        target (GradTarget, optional): what to set on var. Defaults to "closure".


    References:
        Yurii Nesterov, Vladimir Spokoiny. (2015). Random Gradient-Free Minimization of Convex Functions. https://gwern.net/doc/math/2015-nesterov.pdf
    """
    def __init__(
        self,
        h: float = 1e-2,
        n_samples: int = 100,
        formula: _FD_Formula = "forward2",
        distribution: Distributions = "gaussian",
        pre_generate: bool = True,
        return_approx_loss: bool = False,
        target: GradTarget = "closure",
        seed: int | None | torch.Generator = None,
    ):
        super().__init__(h=h, n_samples=n_samples,formula=formula,distribution=distribution,pre_generate=pre_generate,target=target,seed=seed, return_approx_loss=return_approx_loss)

class MeZO(GradApproximator):
    """Gradient approximation via memory-efficient zeroth order optimizer (MeZO) - https://arxiv.org/abs/2305.17333.

    Note:
        This module is a gradient approximator. It modifies the closure to evaluate the estimated gradients,
        and further closure-based modules will use the modified closure. All modules after this will use estimated gradients.

    Args:
        h (float, optional): finite difference step size of jvp_method is set to `forward` or `central`. Defaults to 1e-3.
        n_samples (int, optional): number of random gradient samples. Defaults to 1.
        formula (_FD_Formula, optional): finite difference formula. Defaults to 'central2'.
        distribution (Distributions, optional): distribution. Defaults to "rademacher".
            If this is set to a value higher than zero, instead of using directional derivatives in a new random direction on each step, the direction changes gradually with momentum based on this value. This may make it possible to use methods with memory. Defaults to 0.
        target (GradTarget, optional): what to set on var. Defaults to "closure".

    References:
        Malladi, S., Gao, T., Nichani, E., Damian, A., Lee, J. D., Chen, D., & Arora, S. (2023). Fine-tuning language models with just forward passes. Advances in Neural Information Processing Systems, 36, 53038-53075. https://arxiv.org/abs/2305.17333
    """

    def __init__(self, h: float=1e-3, n_samples: int = 1, formula: _FD_Formula = 'central2',
                 distribution: Distributions = 'rademacher', return_approx_loss: bool = False, target: GradTarget = 'closure'):

        defaults = dict(h=h, formula=formula, n_samples=n_samples, distribution=distribution)
        super().__init__(defaults, return_approx_loss=return_approx_loss, target=target)

    def _seeded_perturbation(self, params: list[torch.Tensor], distribution, seed, h):
        prt = TensorList(params).sample_like(
            distribution=distribution,
            variance=h,
            generator=torch.Generator(params[0].device).manual_seed(seed)
        )
        return prt

    def pre_step(self, objective):
        h = NumberList(self.settings[p]['h'] for p in objective.params)

        n_samples = self.defaults['n_samples']
        distribution = self.defaults['distribution']

        step = objective.current_step

        # create functions that generate a deterministic perturbation from seed based on current step
        prt_fns = []
        for i in range(n_samples):

            prt_fn = partial(self._seeded_perturbation, params=objective.params, distribution=distribution, seed=1_000_000*step + i, h=h)
            prt_fns.append(prt_fn)

        self.global_state['prt_fns'] = prt_fns

    @torch.no_grad
    def approximate(self, closure, params, loss):
        params = TensorList(params)
        loss_approx = None

        h = NumberList(self.settings[p]['h'] for p in params)
        n_samples = self.defaults['n_samples']
        fd_fn = _RFD_FUNCS[self.defaults['formula']]

        prt_fns = self.global_state['prt_fns']

        grad = None
        for i in range(n_samples):
            loss, loss_approx, d = fd_fn(closure=closure, params=params, p_fn=prt_fns[i], h=h, f_0=loss)
            if grad is None: grad = prt_fns[i]().mul_(d)
            else: grad += prt_fns[i]().mul_(d)

        assert grad is not None
        if n_samples > 1: grad.div_(n_samples)
        return grad, loss, loss_approx