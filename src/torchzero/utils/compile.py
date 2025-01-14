"""Experimental and I need to test this on Windows."""
import warnings
import functools
import torch

ENABLE_COMPILING = True

def _try_compiling(warn=False):
    def add(x,y): return x + y
    compled_add = torch.compile(add)
    try:
        res = compled_add(torch.tensor(1.), torch.tensor(2.))
    except Exception as e:
        if warn: warnings.warn(f'Compiling failed so no further functions will be compiled:\n{e}')
        return False
    if res == 3: return True
    return False

class _Compiler:
    def __init__(self, warn=False):
        self.can_compile = None
        self.warn = warn

    def maybe_compile(self, fn, **kwargs):
        if self.can_compile is None: self.can_compile = _try_compiling(self.warn)
        if self.can_compile: return torch.compile(fn, **kwargs)
        return fn

_COMPILER = _Compiler(False)

@functools.wraps(torch.compile)
def maybe_compile(*args, **kwargs):
    """Compiles a function if possible. Same usage as `torch.compile`.

    On first try this will attempt to compile a simple test function. If that fails, all subsequent functions will not be compiled.
    I need to actually test this on windows.
    """
    if ENABLE_COMPILING: return _COMPILER.maybe_compile(*args, **kwargs)
    return args[0]