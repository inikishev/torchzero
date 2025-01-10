r"""
This module includes various basic operators, notable LR for setting the learning rate,
as well as gradient/update clipping and normalization.
"""

from .basic import LR, Clone, Fill, Grad, Identity, Lambda, Zeros
from .on_increase import NegateOnLossIncrease
