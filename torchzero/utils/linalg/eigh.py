import torch
from .utils import mm

def sketched_eigh(
    A_mv: Callable | None,
    A_mm: Callable | None
):
    ...