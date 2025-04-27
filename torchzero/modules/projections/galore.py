import importlib.util
import warnings
from collections.abc import Callable, Mapping
from operator import itemgetter
from typing import Any, Literal

import torch

from ...core import Chainable, Module, Vars
from .projection import Projection


def default_galore_filter(param: torch.Tensor) -> bool:
    return True

def clamp(x, min, max):
    if x < min:return min
    if x > max:return max
    return x

class GaLore(Projection):
    """
    GaLore (Gradient Low-Rank Projection - https://arxiv.org/abs/2403.03507). (NOT IMPLEMENTED FOR TENSORS YET)

    Projects gradients of high-dimensional weight matrices onto a low-rank subspace
    defined by matrices P and Q. This assumes 1st and 2nd dims are in-channels and out-channels.

    Args:
        modules (Chainable):
            Inner optimization module(s) to apply to the projected low-rank gradients (dP, dQ).
        rank (int):
            The rank 'r' for the low-rank projection (per-parameter settings can also use list of ranks per dim).
        update_freq (int, optional):
            Frequency (in steps) for updating the projection matrices P and Q using SVD.
            Defaults to 1 (update every step).
        scale (float, optional):
            Scaling factor applied to the projected gradients before passing them to the
            inner optimizer. Defaults to 1.0.
        layer_filter (Callable[[torch.Tensor], bool], optional):
            A function that takes a parameter tensor and returns True if GaLore should be
            applied to it. The default filter always returns True.
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
        use_tucker: bool = True,
        project_update: bool = True,
        project_params: bool = False,
        project_grad: bool = False,
    ):

        spec = importlib.util.find_spec('tensorly')
        if spec is not None:
            import tensorly # pylint:disable=import-error # pyright:ignore[reportMissingImports]
            self.tensorly = tensorly
            self.tensorly.set_backend('pytorch')
        else:
            self.tensorly = None

        if self.tensorly is None and use_tucker:
            warnings.warn('Tensors will be treated as batched matrices which might be less efficient. You can install tensorly and GaLore will switch to using tucker decomposition for tensors.')

        defaults = dict(
            rank=rank,
            update_freq=update_freq,
            scale=scale,
            layer_filter=layer_filter or default_galore_filter,
            svd_dtype=svd_dtype,
            enable_galore=True, # can be turned off via per-parameter settings
            use_tucker=use_tucker,
        )
        super().__init__(
            modules=modules,
            project_update=project_update,
            project_params=project_params,
            project_grad=project_grad,
            defaults=defaults,
        )

        self.global_state['galore_map'] = {}


    def _should_apply_galore(self, param: torch.Tensor, settings: Mapping[str, Any]) -> tuple[bool,bool]:
        """Check if GaLore should be applied based on filter and dimensions."""
        if not settings['enable_galore']: return False, False
        layer_filter = settings['layer_filter']
        if not layer_filter(param): return False, False

        rank = settings['rank']
        target_ranks = self._get_target_ranks(param.shape, rank)

        use_tucker = settings['use_tucker'] and (self.tensorly is not None) and len([s for s in param.shape if s > 1]) > 2 and \
            all(r < s for r, s in zip(target_ranks, param.shape))

        # False if rank is smaller than dimensions
        if (not use_tucker) and target_ranks[0] >= min(param.shape[0], param.shape[1]): return False, False

        return True, use_tucker

    def _get_target_ranks(self, param_shape: torch.Size, rank: int | list[int]) -> list[int]:
        ndim = len(param_shape)

        if isinstance(rank, int):
            # same same rank for all dims, capped by dimension size
            return [clamp(rank, 1, s-1) for s in param_shape]

        elif isinstance(rank, list):
            if len(rank) == 1 and ndim > 1:
                # same same rank for all dims, capped by dimension size (as length 1 list)
                return [clamp(rank[0], 1, s-1) for s in param_shape]

            elif len(rank) != ndim: # this is manually set per-parameter so raise here
                raise ValueError(f"Length of rank list ({len(rank)}) must match tensor ndim ({ndim}) or be 1.")

            # specified rank per dim, capped by dimension size
            return [clamp(r, 1, s-1) for r, s in zip(rank, param_shape)]

        else:
            raise TypeError(f"Unsupported type for rank: {type(rank)}")


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
            scale = settings['scale']

            # initialize state
            if 'step' not in state:
                state['step'] = 0
                state['galore_applied'] = False
                state['factors_available'] = False

            apply_galore, use_tucker = self._should_apply_galore(param, settings)

            if apply_galore:
                state['galore_applied'] = True
                rank = settings['rank']
                update_freq = settings['update_freq']
                svd_dtype = settings['svd_dtype']
                target_ranks = self._get_target_ranks(param.shape, rank)

                # update P and Q via SVD or factors via tucker
                if state['step'] % update_freq == 0:
                    original_dtype = grad.dtype
                    matrix = grad.to(svd_dtype)
                    try:

                        if use_tucker:
                            assert self.tensorly is not None
                            # based on https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector_tensor.py
                            # need to test different inits?
                            core, factors = self.tensorly.decomposition.tucker(matrix, rank=target_ranks)#, init='random', n_iter_max=1)
                            state['factors'] = [f.to(original_dtype).contiguous() for f in factors]

                        else:
                            needs_transpose = False
                            needs_permute = False
                            if matrix.shape[0] < matrix.shape[1]:
                                matrix = matrix.transpose_(0,1)
                                needs_transpose = True

                            # make sure it is channel last for svd
                            if matrix.ndim > 2:
                                dims = list(range(matrix.ndim))
                                matrix = matrix.permute(*dims[2:], dims[0], dims[1])
                                needs_permute = True

                            rank_svd = target_ranks[0]
                            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False) # pylint:disable=not-callable
                            P = U[:, :rank_svd].to(original_dtype)
                            Q = Vh[:rank_svd, :].mT.to(original_dtype)
                            if needs_transpose: P, Q = Q, P
                            state['P'] = P.contiguous()
                            state['Q'] = Q.contiguous()
                            state['needs_permute'] = needs_permute

                        state['factors_available'] = True

                    except Exception:# as e:#torch.linalg.LinAlgError:
                        # raise e
                        #  warnings.warn(f"SVD failed for parameter {i} with shape {grad.shape}. Skipping GaLore update for this step.", UserWarning)
                        # on fail it will reuse old P and Q that are already in state, I moved the check with tucker check
                        # if 'P' not in state or 'Q' not in state:
                        #     apply_galore = False
                        pass

            # project
            if apply_galore and state['factors_available']:
                if use_tucker:
                    assert self.tensorly is not None
                    factors = state['factors']
                    # project to core tensor: g_core = g x U1.T x U2.T ...
                    g_core = self.tensorly.tenalg.multi_mode_dot(grad, factors, modes=list(range(grad.ndim)), transpose=True) # pylint:disable=no-member # pyright:ignore[reportAttributeAccessIssue]
                    g_core.mul_(scale)
                    projected_gradients_flat.append(g_core)
                    galore_map[i] = {'is_galore': True, 'is_tucker': True, 'indices': flat_idx_counter, 'original_shape': param.shape}
                    flat_idx_counter += 1
                else:
                    P, Q = state['P'], state['Q']
                    gP = P.mT @ grad # (r, m) @ (m, n) -> (r, n)
                    gQ = grad @ Q # (m, n) @ (n, r) -> (m, r)
                    gP.mul_(scale)
                    gQ.mul_(scale)
                    projected_gradients_flat.extend([gP, gQ])
                    galore_map[i] = {
                        "is_galore": True,
                        "is_tucker": False,
                        "indices": (flat_idx_counter, flat_idx_counter + 1),
                        "original_shape": param.shape,
                    }
                    flat_idx_counter += 2

            else:
                # filter, rank issue, or SVD failed without prior P/Q or tucker factors
                projected_gradients_flat.append(grad)
                galore_map[i] = {
                    'is_galore': False,
                    'is_tucker': False,
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

            if map_info['is_galore'] and state['factors_available']:
                if map_info['is_tucker']:
                    assert self.tensorly is not None
                    idx_core = map_info['indices']
                    optim_g_core = tensors[idx_core]
                    factors = state['factors']
                    update = self.tensorly.tenalg.multi_mode_dot(optim_g_core, factors, modes=list(range(param.ndim)) )# pylint:disable=no-member # pyright:ignore[reportAttributeAccessIssue]
                    reconstructed_updates.append(update)
                else:
                    idx_p, idx_q = map_info['indices']
                    optim_gP = tensors[idx_p]
                    optim_gQ = tensors[idx_q]
                    P, Q = state['P'], state['Q']

                    # (m, r) @ (r, n) + (m, r) @ (r, n) -> (m, n) + (m, n) -> (m, n)
                    update = P @ optim_gP + optim_gQ @ Q.mT

                    if state['needs_permute']:
                        dims = list(range(update.ndim))
                        update = update.permute(dims[-2], dims[-1], *dims[:-2])

                    reconstructed_updates.append(update)

            else:
                idx = map_info['indices']
                reconstructed_updates.append(tensors[idx])

                if map_info['is_galore']:
                    # this is probably bad?
                    warnings.warn(f"Unprojecting GaLore parameter {i} that was skipped during projection. Using passthrough value.", UserWarning)

        self.global_state.pop('galore_map', None)
        return reconstructed_updates