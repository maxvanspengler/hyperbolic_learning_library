from math import sqrt
from typing import Optional

from torch import Tensor, empty, matmul, var
from torch.nn.common_types import _size_2_t
from torch.nn.functional import unfold
from torch.nn.init import _calculate_fan_in_and_fan_out, kaiming_uniform_, uniform_

from hypdl.manifolds.base import Manifold
from hypdl.tensors import ManifoldParameter, ManifoldTensor


class Euclidean(Manifold):
    def __init__(self):
        super(Euclidean, self).__init__()

    def project(self, x: ManifoldTensor, dim: int = -1, eps: float = -1.0) -> ManifoldTensor:
        return x

    def expmap0(self, v: ManifoldTensor, dim: int = -1) -> ManifoldTensor:
        return v

    def logmap0(self, y: ManifoldTensor, dim: int = -1) -> ManifoldTensor:
        return y

    def expmap(self, x: ManifoldTensor, v: ManifoldTensor, dim: int = -1) -> ManifoldTensor:
        return x + v

    def logmap(self, x: ManifoldTensor, y: ManifoldTensor, dim: int = -1) -> ManifoldTensor:
        return y - x

    def transp(
        self, x: ManifoldTensor, y: ManifoldTensor, v: ManifoldTensor, dim: int = -1
    ) -> ManifoldTensor:
        return v

    def dist(self, x: ManifoldTensor, y: ManifoldTensor, dim: int = -1) -> ManifoldTensor:
        return (y - x).norm(dim=dim)

    def euc_to_tangent(self, x: ManifoldTensor, u: ManifoldTensor, dim: int = -1) -> ManifoldTensor:
        return u

    def hyperplane_dists(
        self, x: ManifoldTensor, z: ManifoldTensor, r: ManifoldTensor
    ) -> ManifoldTensor:
        if r is None:
            return matmul(x, z)
        else:
            return matmul(x, z) + r

    def fully_connected(
        self, x: ManifoldTensor, z: ManifoldTensor, bias: ManifoldTensor
    ) -> ManifoldTensor:
        if bias is None:
            return matmul(x, z)
        else:
            return matmul(x, z) + bias

    def frechet_mean(self, x: ManifoldTensor, w: Optional[Tensor] = None) -> ManifoldTensor:
        if w is None:
            return x.mean(dim=-1)
        else:
            return matmul(w, x) / w.sum()

    def frechet_variance(
        self, x: Tensor, mu: Tensor, dim: int = -1, w: Optional[Tensor] = None
    ) -> Tensor:
        if w is None:
            return var(x, dim=dim)
        else:
            v1 = w.sum()
            v2 = w.square().sum()
            return matmul(w, (x - mu).square()) / (v1 - v2 / v1)

    def construct_dl_parameters(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> tuple[ManifoldParameter, Optional[ManifoldParameter]]:
        weight = ManifoldParameter(
            data=empty(in_features, out_features),
            manifold=self,
            man_dim=1,
        )

        if bias:
            b = ManifoldParameter(
                data=empty(out_features),
                manifold=self,
                man_dim=1,
            )
        else:
            b = None

        return weight, b

    def reset_parameters(self, weight: ManifoldParameter, bias: ManifoldParameter) -> None:
        kaiming_uniform_(weight, a=sqrt(5))
        if bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(weight)
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            uniform_(bias, -bound, bound)

    def unfold(
        self,
        input: ManifoldTensor,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 1,
        padding: _size_2_t = 0,
        stride: _size_2_t = 1,
    ):
        return unfold(
            input=input,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
