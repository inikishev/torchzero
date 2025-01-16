r"""
This module includes various basic operators, notable LR for setting the learning rate,
as well as gradient/update clipping and normalization.
"""

from .basic import Clone, Fill, Grad, Identity, Lambda, Zeros, Alpha
from .lr import LR
from .on_increase import NegateOnLossIncrease
from .multistep import Multistep
from .accumulate import Accumulate