import pytest
import torch
from torchzero.tensorlist import TensorList

def test_lerp():
    for _ in range(10):
        # out of place
        x = [torch.tensor((4,4), dtype=torch.float32) for _ in range(10)]
        y = [torch.tensor((4,4), dtype=torch.float32) for _ in range(10)]
        
        foreach = torch._foreach_lerp(x, y, 0.33)
        tl = TensorList(x).lerp(y, 0.33)
        tlcompat = TensorList(x).lerp_compat(y, 0.33)
        
        assert all(torch.allclose(i,j) for i,j in zip(foreach, tl))
        assert all(torch.allclose(i,j) for i,j in zip(foreach, tlcompat))
        
        # in-place
        xforeach = [i.clone() for i in x]; yforeach = [i.clone() for i in y]
        xtl = [i.clone() for i in x]; ytl = [i.clone() for i in y]
        xtlcompat = [i.clone() for i in x]; ytlcompat = [i.clone() for i in y]
        torch._foreach_lerp_(xforeach, yforeach, 0.33)
        TensorList(xtl).lerp_(ytl, 0.33)
        TensorList(xtlcompat).lerp_compat_(ytlcompat, 0.33)
        
        assert all(torch.allclose(i,j) for i,j in zip(xforeach, xtl))
        assert all(torch.allclose(i,j) for i,j in zip(xforeach, xtlcompat))