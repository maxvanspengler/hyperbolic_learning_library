import pytest
import torch

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import ManifoldTensor


@pytest.fixture
def manifold_tensor() -> ManifoldTensor:
    return ManifoldTensor(
        data=[
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        manifold=PoincareBall(c=Curvature()),
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
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


def test_slicing(manifold_tensor: ManifoldTensor):
    # First test slicing with the usual numpy slicing
    manifold_tensor = ManifoldTensor(
        data=torch.ones(2, 3, 4, 5, 6, 7, 8, 9),
        manifold=PoincareBall(Curvature()),
        man_dim=-2,
    )

    # Single index
    single_index = manifold_tensor[1]
    assert list(single_index.size()) == [3, 4, 5, 6, 7, 8, 9]
    assert single_index.man_dim == 5

    # More complicated list of slicing arguments
    sliced_tensor = manifold_tensor[1, None, [0, 2], ..., None, 2:5, :, 1]
    # Explanation of the output size: the dimension of size 2 dissappears because of the integer
    # index. Then, a dim of size 1 is added by None. The size 3 dimension reduces to 2 because
    # of the list of indices. The Ellipsis skips the dimensions of sizes 4, 5 and 6. Then another
    # None leads to an insertion of a dimension of size 1. The dimension with size 7 is reduced
    # to 3 because of the 2:5 slice. The manifold dimension of size 8 is left alone and the last
    # dimension is removed because of an integer index.
    assert list(sliced_tensor.size()) == [1, 2, 4, 5, 6, 1, 3, 8]
    assert sliced_tensor.man_dim == 7

    # Now we try to slice into the manifold dimension, which should raise an error
    with pytest.raises(ValueError):
        manifold_tensor[1, None, [0, 2], ..., None, 2:5, 5, 1]

    # Next, we try long tensor indexing, which is used in embeddings
    embedding_manifold_tensor = ManifoldTensor(
        data=torch.ones(10, 3),
        manifold=PoincareBall(Curvature()),
        man_dim=-1,
    )
    indices = torch.Tensor(
        [
            [1, 2],
            [3, 4],
        ]
    ).long()
    embedding_selection = embedding_manifold_tensor[indices]
    assert list(embedding_selection.size()) == [2, 2, 3]
    assert embedding_selection.man_dim == 2

    # This should fail if the man_dim is 0 on the embedding tensor though
    embedding_manifold_tensor = ManifoldTensor(
        data=torch.ones(10, 3),
        manifold=PoincareBall(Curvature()),
        man_dim=0,
    )
    with pytest.raises(ValueError):
        embedding_manifold_tensor[indices]

    # Lastly, embeddings with more dimensions should work too
    embedding_manifold_tensor = ManifoldTensor(
        data=torch.ones(10, 3, 3, 3),
        manifold=PoincareBall(Curvature()),
        man_dim=2,
    )
    embedding_selection = embedding_manifold_tensor[indices]
    assert list(embedding_selection.size()) == [2, 2, 3, 3, 3]
    assert embedding_selection.man_dim == 3


def test_torch_ops(manifold_tensor: ManifoldTensor):
    # We want torch functons to raise an error
    with pytest.raises(TypeError):
        torch.norm(manifold_tensor)

    # Same for torch.Tensor methods (callable attributes)
    with pytest.raises(AttributeError):
        manifold_tensor.mean()
