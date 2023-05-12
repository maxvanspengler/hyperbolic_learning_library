from math import sqrt
from typing import Optional

from torch import Tensor, broadcast_shapes, empty, matmul, var
from torch.nn import Parameter
from torch.nn.common_types import _size_2_t
from torch.nn.functional import unfold
from torch.nn.init import _calculate_fan_in_and_fan_out, kaiming_uniform_, uniform_

from hypdl.manifolds.base import Manifold
from hypdl.tensors import ManifoldParameter, ManifoldTensor, TangentTensor
from hypdl.utils.tensor_utils import (
    check_dims_with_broadcasting,
    check_tangent_tensor_positions,
)


class Euclidean(Manifold):
    def __init__(self):
        super(Euclidean, self).__init__()

    def project(self, x: ManifoldTensor, eps: float = -1.0) -> ManifoldTensor:
        return x

    def expmap(self, v: TangentTensor) -> ManifoldTensor:
        if v.manifold_points is None:
            new_tensor = v.tensor
        else:
            new_tensor = v.manifold_points.tensor + v.tensor
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=v.broadcasted_man_dim)

    def logmap(self, x: Optional[ManifoldTensor], y: ManifoldTensor) -> TangentTensor:
        if x is None:
            dim = y.man_dim
            new_tensor = y.tensor
        else:
            dim = check_dims_with_broadcasting(x, y)
            new_tensor = y.tensor - x.tensor
        return TangentTensor(data=new_tensor, manifold_points=x, manifold=self, man_dim=dim)

    def transp(self, v: TangentTensor, y: ManifoldTensor) -> TangentTensor:
        dim = check_dims_with_broadcasting(v, y)
        output_shape = broadcast_shapes(v.size(), y.size())
        new_tensor = v.tensor.broadcast_to(output_shape)
        return TangentTensor(data=new_tensor, manifold_points=y, manifold=self, man_dim=dim)

    def dist(self, x: ManifoldTensor, y: ManifoldTensor) -> Tensor:
        dim = check_dims_with_broadcasting(x, y)
        return (y.tensor - x.tensor).norm(dim=dim)

    def inner(
        self, u: TangentTensor, v: TangentTensor, keepdim: bool = False, safe_mode: bool = True
    ) -> Tensor:
        dim = check_dims_with_broadcasting(u, v)
        if safe_mode:
            check_tangent_tensor_positions(u, v)

        return (u.tensor * v.tensor).sum(dim=dim, keepdim=keepdim)

    def euc_to_tangent(self, x: ManifoldTensor, u: ManifoldTensor) -> TangentTensor:
        dim = check_dims_with_broadcasting(x, u)
        return TangentTensor(
            data=u.tensor,
            manifold_points=x,
            manifold=self,
            man_dim=dim,
        )

    def hyperplane_dists(self, x: ManifoldTensor, z: ManifoldTensor, r: Optional[Tensor]) -> Tensor:
        if r is None:
            return matmul(x.tensor, z.tensor)
        else:
            return matmul(x.tensor, z.tensor) + r

    def fully_connected(
        self, x: ManifoldTensor, z: ManifoldTensor, bias: Optional[Tensor]
    ) -> ManifoldTensor:
        if bias is None:
            new_tensor = matmul(x.tensor, z.tensor)
        else:
            new_tensor = matmul(x.tensor, z.tensor) + bias
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=-1)

    def frechet_mean(self, x: ManifoldTensor, w: Optional[Tensor] = None) -> ManifoldTensor:
        if w is None:
            mean_tensor = x.tensor.mean(dim=-1)
        else:
            mean_tensor = matmul(w, x.tensor) / w.sum()
        return ManifoldTensor(data=mean_tensor, manifold=self, man_dim=-1)

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
    ) -> tuple[ManifoldParameter, Optional[Parameter]]:
        weight = ManifoldParameter(
            data=empty(in_features, out_features),
            manifold=self,
            man_dim=1,
        )

        if bias:
            b = Parameter(data=empty(out_features))
        else:
            b = None

        return weight, b

    def reset_parameters(self, weight: ManifoldParameter, bias: Parameter) -> None:
        # TODO: check if this actually saves the reset params
        kaiming_uniform_(weight.tensor, a=sqrt(5))
        if bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(weight.tensor)
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            uniform_(bias, -bound, bound)

    def unfold(
        self,
        input: ManifoldTensor,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 1,
        padding: _size_2_t = 0,
        stride: _size_2_t = 1,
    ) -> ManifoldTensor:
        new_tensor = unfold(
            input=input.tensor,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        new_tensor = new_tensor.transpose(1, 2)
        return ManifoldTensor(data=new_tensor, manifold=input.manifold, man_dim=-1)
