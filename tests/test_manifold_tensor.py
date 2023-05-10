import pytest
import torch

from hypdl.manifolds import PoincareBall
from hypdl.tensors import ManifoldTensor


@pytest.fixture
def manifold_tensor() -> ManifoldTensor:
    return ManifoldTensor(
        data=[
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        manifold=PoincareBall(),
        man_dim=-1,
        requires_grad=True,
    )


def test_attributes(manifold_tensor: ManifoldTensor):
    # Check if the standard attributes are set correctly
    # TODO: fix this once __eq__ has been implemented on manifolds
    assert isinstance(manifold_tensor.manifold, PoincareBall)
    assert manifold_tensor.man_dim == 1
    # Check if non-callable attributes are taken from tensor attribute
    assert manifold_tensor.is_cpu


def test_device_methods(manifold_tensor: ManifoldTensor):
    # Check if we can move the manifold tensor to the gpu while keeping it intact
    manifold_tensor = manifold_tensor.cuda()
    assert isinstance(manifold_tensor, ManifoldTensor)
    assert manifold_tensor.is_cuda
    assert isinstance(manifold_tensor.manifold, PoincareBall)
    assert manifold_tensor.man_dim == 1

    # And move it back to the cpu
    manifold_tensor = manifold_tensor.cpu()
    assert isinstance(manifold_tensor, ManifoldTensor)
    assert manifold_tensor.is_cpu
    assert isinstance(manifold_tensor.manifold, PoincareBall)
    assert manifold_tensor.man_dim == 1


def test_torch_ops(manifold_tensor: ManifoldTensor):
    # We want torch functons to raise an error
    with pytest.raises(TypeError):
        torch.norm(manifold_tensor)

    # Same for torch.Tensor methods (callable attributes)
    with pytest.raises(AttributeError):
        manifold_tensor.mean()
