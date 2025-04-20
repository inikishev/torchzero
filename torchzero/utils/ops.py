import torch


def maximum_(input:torch.Tensor, other: torch.Tensor):
    """in-place maximum"""
    return torch.maximum(input, other, out = input)

def where_(input: torch.Tensor, condition: torch.Tensor, other: torch.Tensor):
    """in-place where"""
    return torch.where(condition, input, other, out = input)