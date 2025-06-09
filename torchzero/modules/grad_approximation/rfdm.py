from collections.abc import Callable
from typing import Any
from functools import partial
import torch

from ...utils import TensorList, Distributions, NumberList, generic_eq
from .grad_approximator import GradApproximator, GradTarget, _FD_Formula


def _rforward2(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, v_0: float | None):
    """p_fn is a function that returns the perturbation.
    It may return pre-generated one or generate one deterministically from a seed as in MeZO.
    Returned perturbation must be multiplied by `h`."""
    if v_0 is None: v_0 = closure(False)
    params += p_fn()
    v_plus = closure(False)
    params -= p_fn()
    h = h**2 # because perturbation already multiplied by h
    return v_0, v_0, (v_plus - v_0) / h # (loss, loss_approx, grad)

def _rbackward2(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, v_0: float | None):
    if v_0 is None: v_0 = closure(False)
    params -= p_fn()
    v_minus = closure(False)
    params += p_fn()
    h = h**2 # because perturbation already multiplied by h
    return v_0, v_0, (v_0 - v_minus) / h

def _rcentral2(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, v_0: Any):
    params += p_fn()
    v_plus = closure(False)

    params -= p_fn() * 2
    v_minus = closure(False)

    params += p_fn()
    h = h**2 # because perturbation already multiplied by h
    return v_0, v_plus, (v_plus - v_minus) / (2 * h)

def _rforward3(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, v_0: float | None):
    if v_0 is None: v_0 = closure(False)
    params += p_fn()
    v_plus1 = closure(False)

    params += p_fn()
    v_plus2 = closure(False)

    params -= p_fn() * 2
    h = h**2 # because perturbation already multiplied by h
    return v_0, v_0, (-3*v_0 + 4*v_plus1 - v_plus2) / (2 * h)

def _rbackward3(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, v_0: float | None):
    if v_0 is None: v_0 = closure(False)

    params -= p_fn()
    v_minus1 = closure(False)

    params -= p_fn()
    v_minus2 = closure(False)

    params += p_fn() * 2
    h = h**2 # because perturbation already multiplied by h
    return v_0, v_0, (v_minus2 - 4*v_minus1 + 3*v_0) / (2 * h)

def _rcentral4(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, v_0: float | None):
    params += p_fn()
    v_plus1 = closure(False)

    params += p_fn()
    v_plus2 = closure(False)

    params -= p_fn() * 3
    v_minus1 = closure(False)

    params -= p_fn()
    v_minus2 = closure(False)

    params += p_fn() * 2
    h = h**2 # because perturbation already multiplied by h
    return v_0, v_plus1, (v_minus2 - 8*v_minus1 + 8*v_plus1 - v_plus2) / (12 * h)

_RFD_FUNCS = {
    "forward2": _rforward2,
    "backward2": _rbackward2,
    "central2": _rcentral2,
    "forward3": _rforward3,
    "backward3": _rbackward3,
    "central4": _rcentral4,
}


class RandomizedFDM(GradApproximator):
    """_summary_

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
    """
    PRE_MULTIPLY_BY_H = True
    def __init__(
        self,
        h: float = 1e-3,
        n_samples: int = 1,
        formula: _FD_Formula = "central2",
        distribution: Distributions = "rademacher",
        beta: float = 0,
        pre_generate = True,
        seed: int | None | torch.Generator = None,
        target: GradTarget = "closure",
    ):
        defaults = dict(h=h, formula=formula, n_samples=n_samples, distribution=distribution, beta=beta, pre_generate=pre_generate, seed=seed)
        super().__init__(defaults, target=target)

    def reset(self):
        self.state.clear()
        generator = self.global_state.get('generator', None) # avoid resetting generator
        self.global_state.clear()
        if generator is not None: self.global_state['generator'] = generator

    def _get_generator(self, seed: int | None | torch.Generator, params: list[torch.Tensor]):
        if 'generator' not in self.global_state:
            if isinstance(seed, torch.Generator): self.global_state['generator'] = seed
            elif seed is not None: self.global_state['generator'] = torch.Generator(params[0].device).manual_seed(seed)
            else: self.global_state['generator'] = None
        return self.global_state['generator']

    def pre_step(self, var):
        h, beta = self.get_settings(var.params, 'h', 'beta')
        settings = self.settings[var.params[0]]
        n_samples = settings['n_samples']
        distribution = settings['distribution']
        pre_generate = settings['pre_generate']

        if pre_generate:
            params = TensorList(var.params)
            generator = self._get_generator(settings['seed'], var.params)
            perturbations = [params.sample_like(distribution=distribution, generator=generator) for _ in range(n_samples)]

            if self.PRE_MULTIPLY_BY_H:
                torch._foreach_mul_([p for l in perturbations for p in l], [v for vv in h for v in [vv]*n_samples])

            if all(i==0 for i in beta):
                # just use pre-generated perturbations
                for param, prt in zip(params, zip(*perturbations)):
                    self.state[param]['perturbations'] = prt

            else:
                # lerp old and new perturbations. This makes the subspace change gradually
                # which in theory might improve algorithms with history
                for i,p in enumerate(params):
                    state = self.state[p]
                    if 'perturbations' not in state: state['perturbations'] = [p[i] for p in perturbations]

                cur = [self.state[p]['perturbations'][:n_samples] for p in params]
                cur_flat = [p for l in cur for p in l]
                new_flat = [p for l in zip(*perturbations) for p in l]
                betas = [1-v for b in beta for v in [b]*n_samples]
                torch._foreach_lerp_(cur_flat, new_flat, betas)

    @torch.no_grad
    def approximate(self, closure, params, loss, var):
        params = TensorList(params)
        loss_approx = None

        h = NumberList(self.settings[p]['h'] for p in params)
        settings = self.settings[params[0]]
        n_samples = settings['n_samples']
        fd_fn = _RFD_FUNCS[settings['formula']]
        default = [None]*n_samples
        perturbations = list(zip(*(self.state[p].get('perturbations', default) for p in params)))
        distribution = settings['distribution']
        generator = self._get_generator(settings['seed'], params)

        grad = None
        for i in range(n_samples):
            prt = perturbations[i]
            if prt[0] is None: prt = params.sample_like(distribution=distribution, generator=generator).mul_(h)
            else: prt = TensorList(prt)

            loss, loss_approx, d = fd_fn(closure=closure, params=params, p_fn=lambda: prt, h=h, v_0=loss)
            if grad is None: grad = prt * d
            else: grad += prt * d

        assert grad is not None
        if n_samples > 1: grad.div_(n_samples)
        return grad, loss, loss_approx

SPSA = RandomizedFDM

class RDSA(RandomizedFDM):
    def __init__(
        self,
        h: float = 1e-3,
        n_samples: int = 1,
        formula: _FD_Formula = "central2",
        distribution: Distributions = "gaussian",
        beta: float = 0,
        pre_generate = True,
        target: GradTarget = "closure",
        seed: int | None | torch.Generator = None,
    ):
        super().__init__(h=h, n_samples=n_samples,formula=formula,distribution=distribution,beta=beta,pre_generate=pre_generate,target=target,seed=seed)

class GaussianSmoothing(RandomizedFDM):
    def __init__(
        self,
        h: float = 1e-2,
        n_samples: int = 100,
        formula: _FD_Formula = "central2",
        distribution: Distributions = "gaussian",
        beta: float = 0,
        pre_generate = True,
        target: GradTarget = "closure",
        seed: int | None | torch.Generator = None,
    ):
        super().__init__(h=h, n_samples=n_samples,formula=formula,distribution=distribution,beta=beta,pre_generate=pre_generate,target=target,seed=seed)

class MeZO(GradApproximator):
    def __init__(self, h: float=1e-3, n_samples: int = 1, formula: _FD_Formula = 'central2',
                 distribution: Distributions = 'rademacher', target: GradTarget = 'closure'):
        defaults = dict(h=h, formula=formula, n_samples=n_samples, distribution=distribution)
        super().__init__(defaults, target=target)

    def _seeded_perturbation(self, params: list[torch.Tensor], distribution, seed, h):
        return TensorList(params).sample_like(
            distribution=distribution, generator=torch.Generator(params[0].device).manual_seed(seed)
        ).mul_(h)

    def pre_step(self, var):
        h = NumberList(self.settings[p]['h'] for p in var.params)
        settings = self.settings[var.params[0]]
        n_samples = settings['n_samples']
        distribution = settings['distribution']

        step = var.current_step

        # create functions that generate a deterministic perturbation from seed based on current step
        prt_fns = []
        for i in range(n_samples):

            prt_fn = partial(self._seeded_perturbation, params=var.params, distribution=distribution, seed=1_000_000*step + i, h=h)
            prt_fns.append(prt_fn)

        self.global_state['prt_fns'] = prt_fns

    @torch.no_grad
    def approximate(self, closure, params, loss, var):
        params = TensorList(params)
        loss_approx = None

        h = NumberList(self.settings[p]['h'] for p in params)
        settings = self.settings[params[0]]
        n_samples = settings['n_samples']
        fd_fn = _RFD_FUNCS[settings['formula']]
        prt_fns = self.global_state['prt_fns']

        grad = None
        for i in range(n_samples):
            loss, loss_approx, d = fd_fn(closure=closure, params=params, p_fn=prt_fns[i], h=h, v_0=loss)
            if grad is None: grad = prt_fns[i]().mul_(d)
            else: grad += prt_fns[i]().mul_(d)

        assert grad is not None
        if n_samples > 1: grad.div_(n_samples)
        return grad, loss, loss_approx