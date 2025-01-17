r"""
This includes regularization modules like weight decay.
"""
from .dropout import Dropout
from .noise import AddNoise, Random, add_noise_
from .normalization import (
    Centralize,
    ClipNorm,
    ClipValue,
    Normalize,
    centralize_grad_,
    clip_grad_norm_,
    clip_grad_value_,
    normalize_grad_,
)
from .weight_decay import (
    WeightDecay,
    l1_regularize_,
    l2_regularize_,
    weight_decay_penalty,
)
from .ortho_grad import OrthoGrad, orthograd_