from math import sqrt
from typing import Optional

from torch import Tensor, empty, matmul, var
from torch.nn.common_types import _size_2_t
from torch.nn.functional import unfold
from torch.nn.init import _calculate_fan_in_and_fan_out, kaiming_uniform_, uniform_

from hypdl.manifolds.base import Manifold
from hypdl.tensors import ManifoldParameter, ManifoldTensor
from hypdl.utils.tensor_utils import check_dims_with_broadcasting


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
        new_tensor = x.tensor + v.tensor
        requires_grad = x.requires_grad or v.requires_grad
        return ManifoldTensor(
            data=new_tensor, manifold=x.manifold, man_dim=x.man_dim, requires_grad=requires_grad
        )

    def logmap(self, x: ManifoldTensor, y: ManifoldTensor, dim: int = -1) -> ManifoldTensor:
        new_tensor = y.tensor - x.tensor
        requires_grad = x.requires_grad or y.requires_grad
        return ManifoldTensor(
            data=new_tensor, manifold=x.manifold, man_dim=x.man_dim, requires_grad=requires_grad
        )

    def transp(
        self, x: ManifoldTensor, y: ManifoldTensor, v: ManifoldTensor, dim: int = -1
    ) -> ManifoldTensor:
        return v

    def dist(self, x: ManifoldTensor, y: ManifoldTensor, dim: int = -1) -> Tensor:
        return (y.tensor - x.tensor).norm(dim=dim)

    def inner(
        self, x: ManifoldTensor, u: ManifoldTensor, v: ManifoldTensor, keepdim: bool = False
    ) -> Tensor:
        dim = check_dims_with_broadcasting(x, u, v)
        return (u.tensor * v.tensor).sum(dim=dim, keepdim=keepdim)

    def euc_to_tangent(self, x: ManifoldTensor, u: ManifoldTensor, dim: int = -1) -> ManifoldTensor:
        return u

    def hyperplane_dists(
        self, x: ManifoldTensor, z: ManifoldTensor, r: Optional[ManifoldTensor]
    ) -> Tensor:
        if r is None:
            return matmul(x.tensor, z.tensor)
        else:
            return matmul(x.tensor, z.tensor) + r.tensor

    def fully_connected(
        self, x: ManifoldTensor, z: ManifoldTensor, bias: Optional[ManifoldTensor]
    ) -> Tensor:
        if bias is None:
            return matmul(x.tensor, z.tensor)
        else:
            return matmul(x.tensor, z.tensor) + bias.tensor

    def frechet_mean(self, x: ManifoldTensor, w: Optional[Tensor] = None) -> Tensor:
        if w is None:
            return x.tensor.mean(dim=-1)
        else:
            return matmul(w, x.tensor) / w.sum()

    def frechet_variance(
        self, x: ManifoldTensor, mu: ManifoldTensor, dim: int = -1, w: Optional[Tensor] = None
    ) -> Tensor:
        if w is None:
            return var(x.tensor, dim=dim)
        else:
            v1 = w.sum()
            v2 = w.square().sum()
            return matmul(w, (x.tensor - mu.tensor).square()) / (v1 - v2 / v1)

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
        # TODO: check if this actually saves the reset params
        kaiming_uniform_(weight.tensor, a=sqrt(5))
        if bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(weight.tensor)
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            uniform_(bias.tensor, -bound, bound)

    def unfold(
        self,
        input: ManifoldTensor,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 1,
        padding: _size_2_t = 0,
        stride: _size_2_t = 1,
    ):
        new_tensor = unfold(
            input=input,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        return ManifoldTensor(data=new_tensor, manifold=input.manifold, man_dim=-1)
