import torch

from ...core import Chainable
from ...utils import vec_to_tensors
from ..projections import ProjectionBase


class FFTProjection(ProjectionBase):
    # norm description copied from pytorch docstring
    """Project update into Fourier space of real-valued inputs.

    Args:
        modules (Chainable): modules that will optimize the projected update.
        one_d (bool, optional):
            * If True, uses 1d fft on parameters concatenated into a vector.
            * If False, uses n-dimensional fft on each parameter (default).
        norm (str, optional):
            Normalization mode.

            * "forward" - normalize by 1/n
            * "backward" - no normalization
            * "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)

            Calling the backward transform (:func:`~torch.fft.irfft`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`~torch.fft.irfft`
            the exact inverse.

            Default is "backward" (no normalization).

            The actual torch.fft.rfft default is None, so I set it to None too. I guess None and "backward"
            are the same.
    """

    def __init__(
        self,
        modules: Chainable,
        one_d: bool = False,
        norm=None,
        project_update=True,
        project_params=False,
        project_grad=False,
    ):
        defaults = dict(one_d=one_d, norm=norm)
        super().__init__(modules, project_update=project_update, project_params=project_params, project_grad=project_grad, defaults=defaults)

    @torch.no_grad
    def project(self, tensors, params, grads, loss, states, settings, current):
        settings = settings[0]
        one_d = settings['one_d']
        norm = settings['norm']

        # 1d fft, concatenate all parameters into a vector and calculate fft
        if one_d:
            vec = torch.cat([t.view(-1) for t in tensors])
            self.global_state['length'] = len(vec)
            return [torch.view_as_real(torch.fft.rfft(vec, norm=norm))] # pylint:disable=not-callable

        # multidimensional fft for each parameter
        return [torch.view_as_real(torch.fft.rfftn(t, norm=norm)) if t.numel() > 1 else t for t in tensors] # pylint:disable=not-callable

    @torch.no_grad
    def unproject(self, projected_tensors, params, grads, loss, projected_states, projected_settings, current):
        settings = projected_settings[0]
        one_d = settings['one_d']
        norm = settings['norm']

        if one_d:
            vec = torch.view_as_complex(projected_tensors[0])
            unprojected_vec = torch.fft.irfft(vec, n=self.global_state['length'], norm=norm) # pylint:disable=not-callable
            return vec_to_tensors(unprojected_vec, reference=params)

        return [torch.fft.irfftn(torch.view_as_complex(t.contiguous()), s=p.shape, norm=norm) if t.numel() > 1 else t for t, p in zip(projected_tensors, params)] # pylint:disable=not-callable