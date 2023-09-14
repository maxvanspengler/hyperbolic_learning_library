from math import sqrt
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, broadcast_shapes, empty, matmul, var
from torch.nn import Parameter
from torch.nn.common_types import _size_2_t
from torch.nn.functional import unfold
from torch.nn.init import _calculate_fan_in_and_fan_out, kaiming_uniform_, uniform_

from hypll.manifolds.base import Manifold
from hypll.tensors import ManifoldParameter, ManifoldTensor, TangentTensor
from hypll.utils.tensor_utils import (
    check_dims_with_broadcasting,
    check_if_man_dims_match,
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
        if x.man_dim != 1 or z.man_dim != 0:
            raise ValueError(
                f"Expected the manifold dimension of the inputs to be 1 and the manifold "
                f"dimension of the hyperplane orientations to be 0, but got {x.man_dim} and "
                f"{z.man_dim}, respectively"
            )
        if r is None:
            return matmul(x.tensor, z.tensor)
        else:
            return matmul(x.tensor, z.tensor) + r

    def fully_connected(
        self, x: ManifoldTensor, z: ManifoldTensor, bias: Optional[Tensor]
    ) -> ManifoldTensor:
        if z.man_dim != 0:
            raise ValueError(
                f"Expected the manifold dimension of the hyperplane orientations to be 0, but got "
                f"{z.man_dim} instead"
            )

        dim_shifted_x_tensor = x.tensor.movedim(source=x.man_dim, destination=-1)
        dim_shifted_new_tensor = matmul(dim_shifted_x_tensor, z.tensor)
        if bias is not None:
            dim_shifted_new_tensor = dim_shifted_new_tensor + bias
        new_tensor = dim_shifted_new_tensor.movedim(source=-1, destination=x.man_dim)

        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=x.man_dim)

    def frechet_mean(
        self,
        x: ManifoldTensor,
        batch_dim: Union[int, list[int]] = 0,
        keepdim: bool = False,
    ) -> ManifoldTensor:
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]

        if x.man_dim in batch_dim:
            raise ValueError(
                f"Tried to aggregate over dimensions {batch_dim}, but input has manifold "
                f"dimension {x.man_dim} and cannot aggregate over this dimension"
            )

        # Output manifold dimension is shifted left for each batch dim that disappears
        man_dim_shift = sum(bd < x.man_dim for bd in batch_dim)
        new_man_dim = x.man_dim - man_dim_shift if not keepdim else x.man_dim

        mean_tensor = x.tensor.mean(dim=batch_dim, keepdim=keepdim)

        return ManifoldTensor(data=mean_tensor, manifold=self, man_dim=new_man_dim)

    def midpoint(
        self,
        x: ManifoldTensor,
        batch_dim: Union[int, list[int]] = 0,
        w: Optional[Tensor] = None,
        keepdim: bool = False,
    ) -> ManifoldTensor:
        return self.frechet_mean(x=x, batch_dim=batch_dim, keepdim=keepdim)

    def frechet_variance(
        self,
        x: ManifoldTensor,
        mu: Optional[ManifoldTensor] = None,
        batch_dim: Union[int, list[int]] = -1,
        keepdim: bool = False,
    ) -> Tensor:
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]

        if mu is None:
            return var(x.tensor, dim=batch_dim, keepdim=keepdim)
        else:
            if x.dim() != mu.dim():
                for bd in sorted(batch_dim):
                    mu.man_dim += 1 if bd <= mu.man_dim else 0
                    mu.tensor = mu.tensor.unsqueeze(bd)
            if mu.man_dim != x.man_dim:
                raise ValueError("Input tensor and mean do not have matching manifold dimensions")
            n = 1
            for bd in batch_dim:
                n *= x.size(dim=bd)
            return (x.tensor - mu.tensor).pow(2).sum(dim=batch_dim, keepdim=keepdim) / (n - 1)

    def construct_dl_parameters(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> tuple[ManifoldParameter, Optional[Parameter]]:
        weight = ManifoldParameter(
            data=empty(in_features, out_features),
            manifold=self,
            man_dim=0,
        )

        if bias:
            b = Parameter(data=empty(out_features))
        else:
            b = None

        return weight, b

    def reset_parameters(self, weight: ManifoldParameter, bias: Parameter) -> None:
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
        return ManifoldTensor(data=new_tensor, manifold=input.manifold, man_dim=1)

    def flatten(self, x: ManifoldTensor, start_dim: int = 1, end_dim: int = -1) -> ManifoldTensor:
        """Flattens a manifold tensor by reshaping it. If start_dim or end_dim are passed,
        only dimensions starting with start_dim and ending with end_dim are flattend.

        Updates the manifold dimension if necessary.

        """
        start_dim = x.dim() + start_dim if start_dim < 0 else start_dim
        end_dim = x.dim() + end_dim if end_dim < 0 else end_dim

        # Get the range of dimensions to flatten.
        dimensions_to_flatten = x.shape[start_dim + 1 : end_dim + 1]

        # Get the new manifold dimension.
        if start_dim <= x.man_dim and end_dim >= x.man_dim:
            man_dim = start_dim
        elif end_dim <= x.man_dim:
            man_dim = x.man_dim - len(dimensions_to_flatten)
        else:
            man_dim = x.man_dim

        # Flatten the tensor and return the new instance.
        flattened = torch.flatten(
            input=x.tensor,
            start_dim=start_dim,
            end_dim=end_dim,
        )
        return ManifoldTensor(data=flattened, manifold=x.manifold, man_dim=man_dim)

    def cdist(self, x: ManifoldTensor, y: ManifoldTensor) -> Tensor:
        return torch.cdist(x.tensor, y.tensor)

    def cat(
        self,
        manifold_tensors: Union[Tuple[ManifoldTensor, ...], List[ManifoldTensor]],
        dim: int = 0,
    ) -> ManifoldTensor:
        check_if_man_dims_match(manifold_tensors)
        cat = torch.cat([t.tensor for t in manifold_tensors], dim=dim)
        man_dim = manifold_tensors[0].man_dim
        return ManifoldTensor(data=cat, manifold=self, man_dim=man_dim)
