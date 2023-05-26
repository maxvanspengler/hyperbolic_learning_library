import pytest

from hypdl.manifolds import PoincareBall
from hypdl.nn import HFlatten
from hypdl.tensors import ManifoldTensor


def test_hflatten() -> None:
    manifold_tensor = ManifoldTensor(
        data=[
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ],
        ],
        manifold=PoincareBall(),
        man_dim=1,
        requires_grad=True,
    )

    hflatten = HFlatten(start_dim=1, end_dim=-1)
    flattened = hflatten(manifold_tensor)
    assert flattened.shape == (2, 4)


def test_hflatten__raises_value_error():
    manifold_tensor = ManifoldTensor(
        data=[
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ],
        ],
        manifold=PoincareBall(),
        man_dim=1,
        requires_grad=True,
    )

    hflatten = HFlatten(start_dim=0, end_dim=-1)
    with pytest.raises(ValueError) as e:
        flattened = hflatten(manifold_tensor)
        assert "Can't flatten the manifold dimension 1 of manifold tensor!" in str(e)
