import typing
import torch

from ...core import OptimizerModule

class Cautious(OptimizerModule):
    def __init__(self, normalize = True, eps=1e-6, mode: typing.Literal['zero', 'grad', 'negate'] = 'zero'):
        """Negates update for parameters where update and gradient sign is inconsistent.
        Also normalizes ascent direction by the number of parameters that are not masked.
        This is meant to be used after any momentum-based modules.

        Args:
            normalize (bool, optional):
                renormalize update after masking.
                only has effect when mode is 'zero'. Defaults to True.
            eps (_type_, optional): epsilon for normalization. Defaults to 1e-6.
            mode (str, optional): what to do with updates with inconsistent signs.

                "zero" - set them to zero (as in paper)

                "grad" - set them to the gradient

                "negate" - negate them (same as using update magnitude and gradient sign)

        Note:
            If you use this after modules that estimate gradients, e.g. FDM,
            hey need to have `make_closure` set to True so that they write to `grad` attribute.

        Reference:
            *Cautious Optimizers: Improving Training with One Line of Code.
            Kaizhao Liang, Lizhang Chen, Bo Liu, Qiang Liu*
        """
        super().__init__({})
        self.eps = eps
        self.normalize = normalize
        self.mode = mode

    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()
        grad = state.maybe_compute_grad_(params)

        # mask will be > 0 for parameters where both signs are the same
        mask = (ascent * grad) > 0
        if self.mode in ('zero', 'grad'):
            if self.normalize and self.mode == 'zero':
                fmask = mask.to(ascent[0].dtype)
                fmask /= fmask.total_mean() + self.eps
            else:
                fmask = mask
            ascent *= fmask

            if self.mode == 'grad':
                ascent += grad * mask.logical_not_()

            return ascent

        # mode = 'negate'
        ascent -= ascent.mul(2).mul_(mask)
        return ascent
