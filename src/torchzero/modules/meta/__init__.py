"""Modules that use other modules."""
# from .chain import Chain, ChainReturn
import sys

from .alternate import Alternate
from .grafting import Graft, IntermoduleCautious, SignGrafting
from .return_overrides import ReturnAscent, ReturnClosure, SetGrad

if sys.version_info[1] < 12:
    from .optimizer_wrapper311 import Wrap, WrapClosure
else:
    from .optimizer_wrapper import Wrap, WrapClosure