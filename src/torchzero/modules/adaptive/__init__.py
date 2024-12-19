r"""
Modules related to adapting the learning rate.
"""
from .adaptive import Cautious, UseGradMagnitude, UseGradSign, ScaleLRBySignChange, NegateOnSignChange