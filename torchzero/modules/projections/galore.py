from collections.abc import Callable, Mapping
from typing import Literal, Any
from operator import itemgetter
import warnings

import torch

from ...core import Chainable, Module, Vars
from .projection import Projection

def default_galore_filter(param: torch.Tensor) -> bool:
    return param.ndim >= 2 and param.shape[0] > 1 and param.shape[1] > 1

class GaLore(Projection):
    """
    GaLore (Gradient Low-Rank Projection - https://arxiv.org/abs/2403.03507). (NOT IMPLEMENTED FOR TENSORS YET)

    Projects gradients of high-dimensional weight matrices onto a low-rank subspace
    defined by matrices P and Q. This assumes 1st and 2nd dims are in-channels and out-channels.

    Args:s
        modules (Chainable):
            Inner optimization module(s) to apply to the projected low-rank gradients (dP, dQ).
        rank (int):
            The rank 'r' for the low-rank projection.
        update_freq (int, optional):
            Frequency (in steps) for updating the projection matrices P and Q using SVD.
            Defaults to 1 (update every step).
        scale (float, optional):
            Scaling factor applied to the projected gradients before passing them to the
            inner optimizer. Defaults to 1.0.
        layer_filter (Callable[[torch.Tensor], bool], optional):
            A function that takes a parameter tensor and returns True if GaLore should be
            applied to it. The default filter applies GaLore to
            parameters with ndim >= 2 and size > 1 in the first two dimensions.
        svd_dtype (torch.dtype, optional):
            dtype to use for SVD computation for stability. Defaults to torch.float32.
        project_update (bool): whether to project current update.
        project_params (bool): Whether to project parameters (needed for closures).
        project_grad (bool): Whether to project gradients separately.
    """

    def __init__(
        self,
        modules: Chainable,
        rank: int,
        update_freq: int = 1,
        scale: float = 1.0,
        layer_filter: Callable[[torch.Tensor], bool] = default_galore_filter,
        svd_dtype: torch.dtype = torch.float32,
        project_update: bool = True,
        project_params: bool = False,
        project_grad: bool = False,
    ):

        defaults = dict(
            rank=rank,
            update_freq=update_freq,
            scale=scale,
            layer_filter=layer_filter or default_galore_filter,
            svd_dtype=svd_dtype,
            enable_galore=True # can be turned off via per-parameter settings
        )
        super().__init__(
            modules=modules,
            project_update=project_update,
            project_params=project_params,
            project_grad=project_grad,
            defaults=defaults,
        )

        self.global_state['galore_map'] = {}


    def _should_apply_galore(self, param: torch.Tensor, settings: Mapping[str, Any]) -> bool:
        """Check if GaLore should be applied based on filter and dimensions."""
        if not settings['enable_galore']: return False
        layer_filter = settings['layer_filter']
        rank = settings['rank']
        if not layer_filter(param):
            return False
        # False if rank is smaller than dimensions
        if rank >= min(param.shape[0], param.shape[1]):
            return False
        return True

    @torch.no_grad
    def project(self, tensors, vars):
        """Projects gradients onto low-rank subspaces P and Q."""
        projected_gradients_flat: list[torch.Tensor] = []
        galore_map: dict[int, dict[str, Any]] = {}
        flat_idx_counter = 0
        params_settings = {p: self.settings[p] for p in vars.params}

        for i, (param, grad) in enumerate(zip(vars.params, tensors)):
            settings = params_settings[param]
            state = self.state[param]

            # initialize state
            if 'step' not in state:
                state['step'] = 0
                state['galore_applied'] = False

            apply_galore = self._should_apply_galore(param, settings)

            if apply_galore:
                state['galore_applied'] = True
                rank = settings['rank']
                update_freq = settings['update_freq']
                svd_dtype = settings['svd_dtype']

                # update P and Q via SVD
                if state['step'] % update_freq == 0:
                    original_dtype = grad.dtype
                    matrix = grad.to(svd_dtype)
                    needs_transpose = False
                    if matrix.shape[0] < matrix.shape[1]:
                        matrix = matrix.T
                        needs_transpose = True

                    try:
                        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False) # pylint:disable=not-callable
                        P = U[:, :rank].to(original_dtype)
                        Q = Vh[:rank, :].T.to(original_dtype)

                        if needs_transpose:
                            P, Q = Q, P

                        state['P'] = P.contiguous()
                        state['Q'] = Q.contiguous()
                        state['svd_needs_transpose'] = needs_transpose

                    except torch.linalg.LinAlgError:
                        #  warnings.warn(f"SVD failed for parameter {i} with shape {grad.shape}. Skipping GaLore update for this step.", UserWarning)
                        # on fail it will reuse old P and Q that are already in state
                        if 'P' not in state or 'Q' not in state:
                            apply_galore = False


            if apply_galore and 'P' in state and 'Q' in state:
                P = state['P']
                Q = state['Q']

                # project: gP = P^T @ g, gQ = g @ Q
                gP = P.T @ grad
                gQ = grad @ Q

                # apply scaling
                scale = settings['scale']
                gP.mul_(scale)
                gQ.mul_(scale)

                projected_gradients_flat.extend([gP, gQ])
                galore_map[i] = {
                    'is_galore': True,
                    'indices': (flat_idx_counter, flat_idx_counter + 1),
                    'original_shape': param.shape,
                }
                flat_idx_counter += 2
            else:
                # filter, rank issue, or SVD failed without prior P/Q
                projected_gradients_flat.append(grad)
                galore_map[i] = {
                    'is_galore': False,
                    'indices': flat_idx_counter,
                    'original_shape': param.shape,
                }
                flat_idx_counter += 1

            if state['galore_applied']:
                state['step'] += 1

        self.global_state['galore_map'] = galore_map
        return projected_gradients_flat


    @torch.no_grad
    def unproject(self, tensors, vars):
        """Reconstructs the full update from optimized low-rank projected gradients."""
        reconstructed_updates: list[torch.Tensor] = []
        galore_map = self.global_state.get('galore_map', {})

        for i, param in enumerate(vars.params):
            map_info = galore_map.get(i)
            state = self.state[param]

            if map_info['is_galore'] and state.get('galore_applied', False) and 'P' in state and 'Q' in state:
                idx_p, idx_q = map_info['indices']
                optim_gP = tensors[idx_p]
                optim_gQ = tensors[idx_q]
                P = state['P']
                Q = state['Q']

                # (m, r) @ (r, n) + (m, r) @ (r, n) -> (m, n) + (m, n) -> (m, n)
                update = P @ optim_gP + optim_gQ @ Q.T
                reconstructed_updates.append(update)

            elif map_info['is_galore']:
                idx = map_info['indices']
                reconstructed_updates.append(tensors[idx])
                # this is probably bad?
                warnings.warn(f"Unprojecting GaLore parameter {i} that was skipped during projection. Using passthrough value.", UserWarning)

            else:
                idx = map_info['indices']
                reconstructed_updates.append(tensors[idx])

        self.global_state.pop('galore_map', None)
        return reconstructed_updates