import pytest
import torch
from hypdl.manifolds import PoincareBall
from hypdl.nn import ExpMap0
from hypdl.tensors import ManifoldTensor


def test_expmap0() -> None:
    inputs = torch.randn(10, 2)
    manifold = PoincareBall()
    expmap0 = ExpMap0(
        manifold=manifold,
        man_dim=1,
    )
    manifold_points = expmap0(inputs)
    assert isinstance(manifold_points, ManifoldTensor)
    assert manifold_points.shape == inputs.shape
    assert manifold == manifold_points.manifold
    assert manifold_points.man_dim == 1
