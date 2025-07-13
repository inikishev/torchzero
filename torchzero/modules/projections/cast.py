import torch
from .projection import ProjectionBase
from ...core import Chainable

class To(ProjectionBase):
    """Cast modules to specified device and dtype"""
    def __init__(self, modules: Chainable, dtype: torch.dtype | None, device:torch.types.Device | None = None):
        defaults = dict(dtype=dtype, device=device)
        super().__init__(modules, project_update=True, project_params=True, project_grad=True, defaults=defaults)

    @torch.no_grad
    def project(self, tensors, params, grads, loss, states, settings, current):
        casted = []
        for tensor, state, setting in zip(tensors,states, settings):
            state['dtype'] = tensor.dtype
            state['device'] = tensor.device
            tensor = tensor.to(dtype=setting['dtype'], device=setting['device'])
            casted.append(tensor)
        return casted

    @torch.no_grad
    def unproject(self, projected_tensors, params, grads, loss, states, settings, current):
        uncasted = []
        for tensor, state in zip(projected_tensors, states):
            tensor = tensor.to(dtype=state['dtype'], device=state['device'])
            uncasted.append(tensor)
        return uncasted


class ViewAsReal(ProjectionBase):
    """View complex tensors as real tensors. Doesn't affect tensors that are already."""
    def __init__(self, modules: Chainable):
        super().__init__(modules, project_update=True, project_params=True, project_grad=True, defaults=None)

    @torch.no_grad
    def project(self, tensors, params, grads, loss, states, settings, current):
        views = []
        for tensor, state in zip(tensors,states):
            is_complex = torch.is_complex(tensor)
            state['is_complex'] = is_complex
            if is_complex: tensor = torch.view_as_real(tensor)
            views.append(tensor)
        return views

    @torch.no_grad
    def unproject(self, projected_tensors, params, grads, loss, states, settings, current):
        un_views = []
        for tensor, state in zip(projected_tensors, states):
            if state['is_complex']: tensor = torch.view_as_complex(tensor)
            un_views.append(tensor)
        return un_views
