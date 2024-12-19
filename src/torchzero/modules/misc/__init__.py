r"""
This module includes various basic operators, notable LR for setting the learning rate,
as well as gradient/update clipping and normalization.
"""
from .basic import (LR, Add, AddMagnitude, Clone, Div, Identity, Lambda, Mul,
                    NanToNum, Pow, PowMagnitude, Reciprocal, Sign, sign_grad_)
from .normalization import (Centralize, ClipNorm, ClipValue, Normalize,
                            centralize_grad_, normalize_grad_)
from .on_increase import NegateOnLossIncrease
