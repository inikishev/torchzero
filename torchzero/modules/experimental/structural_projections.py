import math

import numpy as np
import torch

from ...core import Chainable
from ...utils import vec_to_tensors, TensorList
from ..optimizers.shampoo import _merge_small_dims
from ..projections import ProjectionBase



class TensorizeProjection(ProjectionBase):
    """flattens and concatenates all parameters into a vector and then reshapes it into a tensor"""
    def __init__(self, modules: Chainable, max_side: int, project_update=True, project_params=False, project_grad=False):
        defaults = dict(max_side=max_side)
        super().__init__(modules, defaults=defaults, project_update=project_update, project_params=project_params, project_grad=project_grad)

    @torch.no_grad
    def project(self, tensors, params, grads, loss, states, settings, current):
        max_side = self.settings[params[0]]['max_side']
        num_elems = sum(t.numel() for t in tensors)

        if num_elems < max_side:
            self.global_state['remainder'] = 0
            # return 1d
            return [torch.cat([t.view(-1) for t in tensors])]


        # determine appropriate shape to reshape into
        ndims = math.ceil(math.log(num_elems, max_side)) # determine number of dims
        dim_size = math.ceil(num_elems ** (1/ndims)) # average size of a dim with ndims
        dims = [dim_size for _ in range(ndims)]
        required_elems = math.prod(dims)

        # add few extra zeros to vec to match a reshapable size
        remainder = required_elems-num_elems
        if remainder > 0: tensors = tensors + [torch.zeros(remainder, dtype=tensors[0].dtype, device=tensors[0].device)]
        self.global_state['remainder'] = remainder

        # flatten and reshape
        vec = torch.cat([t.view(-1) for t in tensors])
        return [vec.view(dims)]

    @torch.no_grad
    def unproject(self, projected_tensors, params, grads, loss, states, settings, current):
        remainder = self.global_state['remainder']
        # warnings.warn(f'{tensors[0].shape = }')
        vec = projected_tensors[0].view(-1)
        if remainder > 0: vec = vec[:-remainder]
        return vec_to_tensors(vec, params)

class BlockPartition(ProjectionBase):
    """splits parameters into blocks (for now flatttens them and chunks)"""
    def __init__(self, modules: Chainable, max_size: int, batched: bool = False, project_update=True, project_params=False, project_grad=False):
        defaults = dict(max_size=max_size, batched=batched)
        super().__init__(modules, project_update=project_update, project_params=project_params, project_grad=project_grad, defaults=defaults)

    @torch.no_grad
    def project(self, tensors, params, grads, loss, states, settings, current):
        partitioned = []
        for p,t in zip(params, tensors):
            settings = self.settings[p]
            max_size = settings['max_size']
            n = t.numel()
            if n <= max_size:
                partitioned.append(t)
                continue

            t_flat = t.view(-1)

            batched = settings['batched']
            num_chunks = math.ceil(n / max_size)

            if batched:
                chunks_size = num_chunks * max_size
                if num_chunks * max_size > n:
                    t_flat = torch.cat([t_flat, torch.zeros(n-chunks_size, dtype=t_flat.dtype, device=t_flat.device)])
                partitioned.append(t_flat.view(num_chunks, -1))

            else:
                partitioned.extend(t_flat.chunk(num_chunks))

        return partitioned

    @torch.no_grad
    def unproject(self, projected_tensors, params, grads, loss, states, settings, current):
        ti = iter(projected_tensors)
        unprojected = []
        for p in params:
            settings = self.settings[p]
            n = p.numel()

            if settings['batched']:
                unprojected.append(next(ti).view(-1)[:n].view_as(p))

            else:
                chunks = []
                t_n = 0
                while t_n < n:
                    t = next(ti)
                    chunks.append(t)
                    t_n += t.numel()

                assert t_n == n
                unprojected.append(torch.cat(chunks).view_as(p))

        return unprojected

