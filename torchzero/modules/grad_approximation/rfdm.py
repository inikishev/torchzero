from collections.abc import Callable
from typing import Any

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
    def __init__(
        self,
        h: float = 1e-3,
        n_samples: int = 1,
        formula: _FD_Formula = "central2",
        distribution: Distributions = "gaussian",
        beta: float = 0,
        target: GradTarget = "closure",
    ):
        defaults = dict(h=h, formula=formula, n_samples=n_samples, distribution=distribution, beta=beta)
        super().__init__(defaults, target=target)

    def pre_step(self, vars):
        h, beta = self.get_settings('h', 'beta', params=vars.params)
        n_samples = self.defaults['n_samples']
        distribution = self.defaults['distribution']

        if all(i==0 for i in beta):
            # just pre-generate perturbations
            self.global_state['perturbations'] = [
                TensorList(vars.params).sample_like(distribution=distribution).mul_(h) for _ in range(n_samples)
            ]

        else:
            # lerp old and new perturbations. This makes the subspace change gradually
            # which in theory might improve algorithms with history
            perts = self.global_state['perturbations'] = self.global_state.get('perturbations', [])[:n_samples] # trim if n_samples changed
            for i in range(n_samples):
                new_pert = TensorList(vars.params).sample_like(distribution=distribution).mul_(h)
                if i >= len(perts):
                    perts.append(new_pert)
                else:
                    perts[i].lerp_(new_pert, [1-b for b in beta])

    @torch.no_grad
    def approximate(self, closure, params, loss, vars):
        params = TensorList(params)
        loss_approx = None

        h = self.get_settings('h', params=vars.params, cls=NumberList)
        n_samples = self.defaults['n_samples']
        fd_fn = _RFD_FUNCS[self.defaults['formula']]
        perturbations = self.global_state['perturbations']

        grad = None
        for i in range(n_samples):
            prt = perturbations[i]
            loss, loss_approx, d = fd_fn(closure=closure, params=params, p_fn=lambda: prt, h=h, v_0=loss)
            if grad is None: grad = prt * d
            else: grad += prt * d

        assert grad is not None
        return grad, loss, loss_approx


class MeZO(GradApproximator):
    def __init__(self, h: float=1e-3, n_samples: int = 1, formula: _FD_Formula = 'central2',
                 distribution: Distributions = 'gaussian', target: GradTarget = 'closure'):
        defaults = dict(h=h, formula=formula, n_samples=n_samples, distribution=distribution)
        super().__init__(defaults, target=target)

    def pre_step(self, vars):
        h = self.get_settings('h', params=vars.params)
        n_samples = self.defaults['n_samples']
        distribution = self.defaults['distribution']

        step = vars.current_step

        # create functions that generate a deterministic perturbation from seed based on current step
        prt_fns = []
        for i in range(n_samples):

            prt_fn = lambda: TensorList(vars.params).sample_like(
                distribution=distribution, generator=torch.Generator(device=vars.params[0].device)
                .manual_seed(1_000_000*step + i)).mul_(h)

            prt_fns.append(prt_fn)

        self.global_state['prt_fns'] = prt_fns

    @torch.no_grad
    def approximate(self, closure, params, loss, vars):
        params = TensorList(params)
        loss_approx = None

        h = self.get_settings('h', params=vars.params, cls=NumberList)
        n_samples = self.defaults['n_samples']
        fd_fn = _RFD_FUNCS[self.defaults['formula']]
        prt_fns = self.global_state['prt_fns']

        grad = None
        for i in range(n_samples):
            loss, loss_approx, d = fd_fn(closure=closure, params=params, p_fn=prt_fns[i], h=h, v_0=loss)
            if grad is None: grad = prt_fns[i]().mul_(d)
            else: grad += prt_fns[i]().mul_(d)

        assert grad is not None
        return grad, loss, loss_approx