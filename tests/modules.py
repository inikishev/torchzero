from typing import Any
import pytest
import torch
import torchzero as tz

def test_cautious():
    p = torch.tensor([1., 1.], requires_grad = True)
    p.grad = torch.tensor([0.2, 0.2])
