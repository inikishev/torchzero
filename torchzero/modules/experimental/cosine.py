"""A bunch of useless modules that I hate and that didn't work"""
import torch

from ...core import Chainable, Transform, apply_transform
from ...utils import NumberList, TensorList, as_tensorlist, unpack_dicts, unpack_states


class CosineStepSize(Transform):
    """Adaptive step size based on cosine similarity

    VERDICT: Useless. This is too unstable.

    Args:
        scale (float, optional): cosine similarity multiplier. Defaults to 0.95.
        init (float, optional): initial step size. Defaults to 1.
        eps (float, optional): epsilon for division stability. Defaults to 1e-12.
        target_cossim (float, optional): cosine similarity needs to be above this to increase step size. Defaults to 1e-8.
        inner (Chainable | None, optional):
            inner modules applied after calculating cosine similarity and before step size correction. Defaults to None.
    """
    def __init__(self, scale:float = 0.95, init:float=1, eps:float=1e-12, inner:Chainable | None = None):
        defaults = dict(scale=scale, init=init, eps=eps)
        super().__init__(defaults, uses_grad=False)
        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        scale, init = unpack_dicts(settings, 'scale', 'init', cls=NumberList)
        unpack_states(states, tensors, 'alpha', init=init, cls=NumberList) # initializes alpha to init
        eps = settings[0]['eps']

        tensors = as_tensorlist(tensors)
        prev = unpack_states(states, tensors, 'prev', init=tensors, cls=TensorList)

        tensors_norm = tensors.global_vector_norm()
        cos_sim = (tensors.dot(prev) / (tensors_norm * prev.global_vector_norm()).clip(min=eps)).item()

        if 'inner' in self.children:
            tensors = as_tensorlist(apply_transform(self.children['inner'], tensors, params, grads, loss))

        new_alpha = []
        for s, sc in zip(states, scale):
            s['alpha'] *= 1 + cos_sim * sc
            new_alpha.append(s['alpha'])

        tensors.mul_(new_alpha)
        prev.copy_(tensors)

        return tensors



class CosineDebounce(Transform):
    """Debouncing when cosine similarity is less than 0.

    VERDICT: Useless. This doesn't help at all.

    Args:
        scale (float, optional): cosine similarity multiplier. Defaults to 0.95.
        eps (float, optional): epsilon for division stability. Defaults to 1e-12.
        inner (Chainable | None, optional):
            inner modules applied after calculating cosine similarity and before debouncing correction. Defaults to None.
    """
    def __init__(self, scale:float = 0.95, eps:float=1e-12, damping:float=0.95, inner:Chainable | None = None):
        defaults = dict(scale=scale, eps=eps, damping=damping)
        super().__init__(defaults, uses_grad=False)
        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        scale, damping = unpack_dicts(settings, 'scale', 'damping', cls=NumberList)
        eps = settings[0]['eps']

        tensors = as_tensorlist(tensors)
        prev = unpack_states(states, tensors, 'prev', init=tensors, cls=TensorList).mul_(damping)

        tensors_norm = tensors.global_vector_norm()
        cos_sim = (tensors.dot(prev) / (tensors_norm * prev.global_vector_norm()).clip(min=eps)).item()

        if 'inner' in self.children:
            tensors = as_tensorlist(apply_transform(self.children['inner'], tensors, params, grads, loss))

        if cos_sim < -eps:
            undo = prev.neg().mul_(-cos_sim * scale)
            comb = prev.graft(tensors).add_(tensors).graft_(prev).mul_(-cos_sim*scale)
            tensors = undo.add_(comb)

        prev.copy_(tensors)
        return tensors



class CosineMomentum(Transform):
    """Beta depends on cosine similarity. At cossim=1, beta is 0. At cossim=-1, beta is 2^power. This basically removes oscillations.

    VERDICT: Useless. Worse than all other momentums.

    Args:
        scale (float, optional): cosine similarity multiplier. Defaults to 1.
        nesterov (float, optional): whether to use nesterov momentum. Defaults to False.
        power (float, optional): power for beta. Defaults to 1.
        eps (float, optional): epsilon for division stability. Defaults to 1e-12.
        inner (Chainable | None, optional):
            inner modules applied after calculating cosine similarity and before updating exponential moving average. Defaults to None.
    """
    def __init__(self, scale:float = 1, nesterov: bool = False, power: float = 1, eps:float=1e-12, inner:Chainable | None = None):
        defaults = dict(scale=scale, eps=eps, nesterov=nesterov, power=power)
        super().__init__(defaults, uses_grad=False)
        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        scale, power = unpack_dicts(settings, 'scale', 'power', cls=NumberList)
        eps = settings[0]['eps']
        nesterov = settings[0]['nesterov']
        exp_avg = unpack_states(states, tensors, 'exp_avg', cls=TensorList)

        tensors = as_tensorlist(tensors)

        tensors_norm = tensors.global_vector_norm()
        cos_sim = (tensors.dot(exp_avg) / (tensors_norm * exp_avg.global_vector_norm()).clip(min=eps)).item()

        if 'inner' in self.children:
            tensors = as_tensorlist(apply_transform(self.children['inner'], tensors, params, grads, loss))

        beta = (1 - (cos_sim*scale)) ** power
        if nesterov:
            exp_avg.add_(tensors.mul(beta))
            return tensors.add_(exp_avg)
        else:
            exp_avg.add_(tensors.mul_(beta))
            return exp_avg.clone()


class AdaptiveDifference(Transform):
    """VERDICT: Useless. Doesn't help (sort of to be expected)."""
    def __init__(self, inner:Chainable | None = None):
        defaults = dict()
        super().__init__(defaults, uses_grad=False)
        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        tensors = as_tensorlist(tensors)
        prev = unpack_states(states, tensors, 'prev', init=tensors, cls=TensorList)

        diff = tensors - prev.graft_(tensors)
        prev.copy_(tensors)

        if 'inner' in self.children:
            tensors = as_tensorlist(apply_transform(self.children['inner'], tensors, params, grads, loss))

        tensors.add_(diff.graft_(tensors))

        return tensors

class AdaptiveDifferenceEMA(Transform):
    """VERDICT: better than non-EMA but still useless."""
    def __init__(self, beta=0.99, inner:Chainable | None = None):
        defaults = dict(beta=beta)
        super().__init__(defaults, uses_grad=False)
        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        tensors = as_tensorlist(tensors)
        beta = unpack_dicts(settings, 'beta', cls=NumberList)
        prev, diff_exp_avg = unpack_states(states, tensors, 'prev', 'diff_exp_avg', init=[tensors,torch.zeros_like], cls=TensorList)

        diff = (tensors - prev.graft_(tensors)).graft_(tensors)
        diff_exp_avg.lerp_(diff, 1-beta)
        prev.copy_(tensors)

        if 'inner' in self.children:
            tensors = as_tensorlist(apply_transform(self.children['inner'], tensors, params, grads, loss))

        tensors.add_(diff_exp_avg.graft(tensors))

        return tensors


class ScaledAdaptiveDifference(Transform):
    """VERDICT: Useless and doesn't help."""
    def __init__(self, scale=0.95, damping:float=0.99, inner:Chainable | None = None):
        defaults = dict(scale=scale, damping=damping)
        super().__init__(defaults, uses_grad=False)
        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        tensors = as_tensorlist(tensors)
        scale, damping = unpack_dicts(settings, 'scale', 'damping', cls=NumberList)
        prev_tensors, prev_update = unpack_states(states, tensors, 'prev', 'prev_update', init=[tensors,tensors], cls=TensorList)

        cos_sim = (tensors.dot(prev_update) / (tensors.global_vector_norm() * prev_update.global_vector_norm()).clip(min=1e-10)).item()

        if 'inner' in self.children:
            tensors = as_tensorlist(apply_transform(self.children['inner'], tensors, params, grads, loss))

        if cos_sim > 0:
            tensors.add_(prev_tensors*(cos_sim*scale))

        else:
            undo = prev_tensors.neg().mul_(-cos_sim*scale)
            comb = prev_tensors.graft(tensors).add_(tensors).graft_(prev_tensors).mul_(-cos_sim*scale)
            tensors = undo.add_(comb).graft_((tensors-prev_tensors).mul_(damping))

        diff = tensors - prev_tensors.graft_(tensors)
        prev_tensors.copy_(tensors)
        diff.graft_(tensors)
        tensors.add_(diff)
        prev_update.copy_(tensors)

        return tensors
