import importlib.util
import warnings
from collections.abc import Callable, Mapping
from operator import itemgetter
from typing import Any, Literal

import torch

from ...core import Chainable, Module, Objective
from .projection import ProjectionBase

# TODO