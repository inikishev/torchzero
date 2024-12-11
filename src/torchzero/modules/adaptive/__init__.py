r"""
Modules related to adapting the learning rate.
"""
from .cautious import Cautious
from .sign_change import ScaleLRBySignChange, NegateOnSignChange