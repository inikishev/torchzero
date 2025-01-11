# pylint: disable = singleton-comparison
# ruff: noqa: E712
r"""
This submodule contains composable optimizer "building blocks".
"""
from collections.abc import Iterable, Sequence
from typing import Any, Literal, TypedDict, get_type_hints

from ..core.module import OptimizerModule
from .adaptive import *
from .gradient_approximation import *
from .line_search import *
from .meta import *
from .misc import *
from .momentum import *
from .operations import *
from .optimizers import *
from .quasi_newton import *
from .regularization import *
from .second_order import *
from .smoothing import *
from . import experimental
# from .experimental.subspace import *
