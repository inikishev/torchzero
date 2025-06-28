import torch

from ...core import Transform, Chainable, apply_transform
from ...utils import TensorList, NumberList, unpack_dicts, unpack_states, as_tensorlist


class CosineTrustRegion(Transform):
    """Trust region plus a kind of debouncing based on cosine similarity.

    When current update points in same direction as previous update, step size is increased.
    Otherwise step size is decreased, previous step is undone, and a scaled combination of previous and current step is made.

    Args:
        scale (float, optional): cosine similarity scale. Defaults to 0.95.
        init (float, optional): initial step size. Defaults to 1.
        eps (float, optional): epsilon for division stability. Defaults to 1e-12.
        target_cossim (float, optional): cosine similarity needs to be above this to increase step size. Defaults to 1e-8.
        inner (Chainable | None, optional): inner modules. Defaults to None.
    """
    def __init__(self, scale:float = 0.95, init:float=1, eps:float=1e-12, stopping:float=0.99, inner:Chainable | None = None):
        defaults = dict(scale=scale, init=init, eps=eps, stopping=stopping)
        super().__init__(defaults, uses_grad=False)
        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        scale, init, stopping = unpack_dicts(settings, 'scale', 'init', 'stopping', cls=NumberList)
        eps = settings[0]['eps']

        tensors = as_tensorlist(tensors)
        alpha = unpack_states(states, tensors, 'alpha', init=init, cls=NumberList)
        prev = unpack_states(states, tensors, 'prev', init=tensors, cls=TensorList)
        skip = self.global_state.get('skip', False)
        self.global_state['skip'] = False

        tensors_norm = tensors.global_vector_norm()
        cos_sim = (tensors.dot(prev) / (tensors_norm * prev.global_vector_norm()).clip(min=eps)).item()

        if 'inner' in self.children:
            tensors = as_tensorlist(apply_transform(self.children['inner'], tensors, params, grads, loss))

        if cos_sim < -eps:
            self.global_state['skip'] = True
            if skip: prev.mul_(stopping)
            skip = False

        if skip or tensors_norm < eps:
            tensors.mul_(alpha)
            prev.copy_(tensors)
            self.global_state['skip'] = False
            return tensors

        new_alpha = []
        for s, sc in zip(states, scale):
            s['alpha'] *= 1 + cos_sim * sc
            new_alpha.append(s['alpha'])

        if cos_sim < -eps:
            undo = prev.neg().mul_(-cos_sim * scale)
            comb = prev.graft(tensors).add_(tensors).graft_(prev).mul_(-cos_sim*scale)
            tensors = undo.add_(comb)

        else:
            tensors.mul_(new_alpha)

        prev.copy_(tensors)

        return tensors



class CosineDebounce(Transform):
    """Debouncing when cosine similarity is less than 0.

    Args:
        scale (float, optional): cosine similarity scale. Defaults to 0.95.
        eps (float, optional): epsilon for division stability. Defaults to 1e-12.
        inner (Chainable | None, optional): inner modules. Defaults to None.
    """
    def __init__(self, scale:float = 0.95, eps:float=1e-12, damping:float=0.95, inner:Chainable | None = None):
        defaults = dict(scale=scale, eps=eps, damping=damping)
        super().__init__(defaults, uses_grad=False)
        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
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

