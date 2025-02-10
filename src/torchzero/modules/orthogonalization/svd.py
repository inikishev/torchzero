"""Orthogonalization code adapted from https://github.com/MarkTuddenham/Orthogonal-Optimisers

Tuddenham, M., Prügel-Bennett, A., & Hare, J. (2022).
Orthogonalising gradients to speed up neural network optimisation. arXiv preprint arXiv:2202.07052.
"""
import logging
from collections.abc import Iterable, Sequence

import torch

from ...core import OptimizerModule, _Targets

@torch.no_grad()
def _orthogonalize_update_(updates: Sequence[torch.Tensor], toggle = None, warn_fail=True) -> None:
    """adapted from https://github.com/MarkTuddenham/Orthogonal-Optimisers"""
    if toggle is None: toggle = [True] * len(updates)

    # Orthogonalise the gradients using SVD
    for grad, orth in zip(updates, toggle):
        if orth and grad.ndim > 1:
            G: torch.Tensor = grad.view(grad.shape[0], -1)
            orth_G: torch.Tensor | None = None
            try:
                u, s, vt = torch.linalg.svd(G, full_matrices=False) # pylint:disable=not-callable
                orth_G = u @ vt
            except RuntimeError:
                # if warn: logging.warning('Failed to perform SVD, adding some noise.')
                try:
                    u, s, v = torch.svd_lowrank(
                        G,
                        q=1,    # assume rank is at least 1
                        M=1e-4 * G.mean() * torch.randn_like(G))
                    orth_G = u @ v.T
                except RuntimeError:
                    if warn_fail: logging.error(('Failed to perform SVD with noise,'
                                    ' skipping gradient orthogonalisation'))
            if orth_G is not None:
                grad.set_(orth_G.reshape_as(grad)) # type:ignore

    return updates

def orthogonalize_grad_(params: Iterable[torch.Tensor], warn_fail=False):
    """orthogonalizes gradients of an iterable of parameters.

    This updates gradients in-place.

    The orthogonalization code is adapted from https://github.com/MarkTuddenham/Orthogonal-Optimisers
    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to orthogonalize.
        warn_fail (bool, optional):
            whether to print a warning when orthogonalization fails, and gradients are not
            orthogonalized. Defaults to True.
    """
    grads = [p.grad for p in params if p.grad is not None]
    _orthogonalize_update_(grads, warn_fail=warn_fail)

class Orthogonalize(OptimizerModule):
    """Orthogonalizes the update using SVD.

    To disable orthogonalization for a parameter, put it into a parameter group with "orth" = False.

    The orthogonalization code is adapted from https://github.com/MarkTuddenham/Orthogonal-Optimisers

    Tip: :py:class:`tz.m.ZeropowerViaNewtonSchulz` is a significantly faster version of this.
    Args:
        warn_fail (bool, optional):
            whether to print a warning when orthogonalization fails, and gradients are not
            orthogonalized. Defaults to True.
        target (str, optional):
            determines what this module updates.

            "ascent" - it updates the ascent

            "grad" - it updates the gradient (and sets `.grad` attributes to updated gradient).

            "closure" - it makes a new closure that sets the updated ascent to the .`grad` attributes.
    """
    def __init__(self, warn_fail=True, target: _Targets = 'ascent'):
        defaults = dict(orth = True)
        super().__init__(defaults, target = target)
        self.warn_fail = warn_fail

    def _update(self, vars, ascent):
        toggle = self.get_group_key('orth', cls=list)
        _orthogonalize_update_(ascent, toggle, self.warn_fail)
        return ascent