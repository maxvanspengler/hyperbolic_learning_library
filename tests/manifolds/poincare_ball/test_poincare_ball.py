import torch

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import ManifoldTensor


def test_flatten__man_dim_equals_start_dim() -> None:
    tensor = torch.randn(2, 2, 2, 2)
    manifold = PoincareBall(c=Curvature())
    manifold_tensor = ManifoldTensor(
        data=tensor,
        manifold=manifold,
        man_dim=1,
        requires_grad=True,
    )
    flattened = manifold.flatten(manifold_tensor, start_dim=1, end_dim=-1)
    assert flattened.shape == (2, 8)
    assert flattened.man_dim == 1


def test_flatten__man_dim_equals_start_dim__set_end_dim() -> None:
    tensor = torch.randn(2, 2, 2, 2)
    manifold = PoincareBall(c=Curvature())
    manifold_tensor = ManifoldTensor(
        data=tensor,
        manifold=manifold,
        man_dim=1,
        requires_grad=True,
    )
    flattened = manifold.flatten(manifold_tensor, start_dim=1, end_dim=2)
    assert flattened.shape == (2, 4, 2)
    assert flattened.man_dim == 1


def test_flatten__man_dim_larger_start_dim() -> None:
    tensor = torch.randn(2, 2, 2, 2)
    manifold = PoincareBall(c=Curvature())
    manifold_tensor = ManifoldTensor(
        data=tensor,
        manifold=manifold,
        man_dim=2,
        requires_grad=True,
    )
    flattened = manifold.flatten(manifold_tensor, start_dim=1, end_dim=-1)
    assert flattened.shape == (2, 8)
    assert flattened.man_dim == 1


def test_flatten__man_dim_larger_start_dim__set_end_dim() -> None:
    tensor = torch.randn(2, 2, 2, 2)
    manifold = PoincareBall(c=Curvature())
    manifold_tensor = ManifoldTensor(
        data=tensor,
        manifold=manifold,
        man_dim=2,
        requires_grad=True,
    )
    flattened = manifold.flatten(manifold_tensor, start_dim=1, end_dim=2)
    assert flattened.shape == (2, 4, 2)
    assert flattened.man_dim == 1


def test_flatten__man_dim_smaller_start_dim() -> None:
    tensor = torch.randn(2, 2, 2, 2)
    manifold = PoincareBall(c=Curvature())
    manifold_tensor = ManifoldTensor(
        data=tensor,
        manifold=manifold,
        man_dim=0,
        requires_grad=True,
    )
    flattened = manifold.flatten(manifold_tensor, start_dim=1, end_dim=-1)
    assert flattened.shape == (2, 8)
    assert flattened.man_dim == 0


def test_flatten__man_dim_larger_end_dim() -> None:
    tensor = torch.randn(2, 2, 2, 2)
    manifold = PoincareBall(c=Curvature())
    manifold_tensor = ManifoldTensor(
        data=tensor,
        manifold=manifold,
        man_dim=2,
        requires_grad=True,
    )
    flattened = manifold.flatten(manifold_tensor, start_dim=0, end_dim=1)
    assert flattened.shape == (4, 2, 2)
    assert flattened.man_dim == 1


def test_cdist__correct_dist():
    B, P, R, M = 2, 3, 4, 8
    manifold = PoincareBall(c=Curvature())
    mt1 = ManifoldTensor(torch.randn(B, P, M), manifold=manifold)
    mt2 = ManifoldTensor(torch.randn(B, R, M), manifold=manifold)
    dist_matrix = manifold.cdist(mt1, mt2)
    for b in range(B):
        for p in range(P):
            for r in range(R):
                assert torch.isclose(
                    dist_matrix[b, p, r], manifold.dist(mt1[b, p], mt2[b, r]), equal_nan=True
                )


def test_cdist__correct_dims():
    B, P, R, M = 2, 3, 4, 8
    manifold = PoincareBall(c=Curvature())
    mt1 = ManifoldTensor(torch.randn(B, P, M), manifold=manifold)
    mt2 = ManifoldTensor(torch.randn(B, R, M), manifold=manifold)
    dist_matrix = manifold.cdist(mt1, mt2)
    assert dist_matrix.shape == (B, P, R)


def test_cat__correct_dims():
    N, D1, D2 = 10, 2, 3
    manifold = PoincareBall(c=Curvature())
    manifold_tensors = [ManifoldTensor(torch.randn(D1, D2), manifold=manifold) for _ in range(N)]
    cat_0 = manifold.cat(manifold_tensors, dim=0)
    cat_1 = manifold.cat(manifold_tensors, dim=1)
    assert cat_0.shape == (D1 * N, D2)
    assert cat_1.shape == (D1, D2 * N)


def test_cat__correct_man_dim():
    N, D1, D2 = 10, 2, 3
    manifold = PoincareBall(c=Curvature())
    manifold_tensors = [
        ManifoldTensor(torch.randn(D1, D2), manifold=manifold, man_dim=1) for _ in range(N)
    ]
    cat_0 = manifold.cat(manifold_tensors, dim=0)
    cat_1 = manifold.cat(manifold_tensors, dim=1)
    assert cat_0.man_dim == cat_1.man_dim == 1


def test_cat__beta_concatenation_correct_norm():
    MD = 64
    manifold = PoincareBall(c=Curvature(0.1, constraining_strategy=lambda x: x))
    t1 = torch.randn(MD)
    t2 = torch.randn(MD)
    mt1 = ManifoldTensor(t1 / t1.norm(), manifold=manifold)
    mt2 = ManifoldTensor(t2 / t2.norm(), manifold=manifold)
    cat = manifold.cat([mt1, mt2])
    assert torch.isclose(cat.tensor.norm(), torch.as_tensor(1.0), atol=1e-2)
