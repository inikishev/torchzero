r"""
This module includes various basic operators, notable LR for setting the learning rate,
as well as gradient/update clipping and normalization.
"""
from .basic import Clone, Lambda, Reciprocal, NanToNum, Identity, Mul, Div, Add, AddMagnitude, Pow, PowMagnitude
from .clipping import ClipNorm, ClipValue
from .lr import LR
from .normalization import Normalize, normalize_grad_
from .on_increase import NegateOnLossIncrease
from .sign import Sign, sign_grad_