import torch

# set to false by default for windows support
# how do I make it true for my system without forgetting to set to false on push???
COMPILE = False

# also I need to actually use this
def maybe_compile(func):
    if COMPILE: return torch.compile(func)
    return func