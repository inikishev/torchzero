r"""
This includes regularization modules like weight decay.
"""
from .weight_decay import l1_regularize_, l2_regularize_, weight_decay_penalty, WeightDecay
from .noise import AddNoise
from .dropout import Dropout
from .lr_scaling import ReduceOutwardLR
