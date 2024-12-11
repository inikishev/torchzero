r"""
Modules related to adapting the learning rate.
"""
from .cautious import Cautious, NegateOnSignChange, NegateOnSignInconsistence, UndoOnSignChange
from .rprop import Rprop