from typing import Optional

import torch.nn as nn
from torch import Tensor, as_tensor, empty, eye, float32, no_grad
from torch.nn.common_types import _size_2_t
from torch.nn.functional import unfold
from torch.nn.init import normal_, zeros_

from hypdl.manifolds.base import Manifold
from hypdl.manifolds.euclidean import Euclidean
from hypdl.tensors import ManifoldParameter, ManifoldTensor
from hypdl.utils.math import beta_func
from hypdl.utils.tensor_utils import check_dims_with_broadcasting

from .math.diffgeom import (
    dist,
    euc_to_tangent,
    expmap,
    expmap0,
    gyration,
    inner,
    logmap,
    logmap0,
    mobius_add,
    project,
    transp,
)
from .math.linalg import poincare_fully_connected, poincare_hyperplane_dists
from .math.stats import frechet_mean, frechet_variance


class PoincareBall(Manifold):
    """
    Class representing the Poincare ball model of hyperbolic space.

    Implementation based on the geoopt implementation,
    but changed to use hyperbolic torch functions.
    """

    def __init__(self, c=1.0, learnable=False):
        super(PoincareBall, self).__init__()
        c = as_tensor(c, dtype=float32)
        self.isp_c = nn.Parameter(c, requires_grad=learnable)
        self.learnable = learnable

    @property
    def c(self) -> Tensor:
        # TODO: should probably inverse softplus during init to account for this
        # otherwise setting c = 1.0 doesn't actually lead to ball.c = 1.0
        return nn.functional.softplus(self.isp_c)

    def mobius_add(self, x: ManifoldTensor, y: ManifoldTensor) -> ManifoldTensor:
        dim = check_dims_with_broadcasting(x, y)
        new_tensor = mobius_add(x=x, y=y, c=self.c, dim=dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=dim)

    def project(self, x: ManifoldTensor, eps: float = -1.0) -> ManifoldTensor:
        new_tensor = project(x=x.tensor, c=self.c, dim=x.man_dim, eps=eps)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=x.man_dim)

    def expmap0(self, v: ManifoldTensor) -> ManifoldTensor:
        new_tensor = expmap0(v=v.tensor, c=self.c, dim=v.man_dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=v.man_dim)

    def logmap0(self, y: ManifoldTensor) -> ManifoldTensor:
        new_tensor = logmap0(y=y.tensor, c=self.c, dim=y.man_dim)
        return ManifoldTensor(data=new_tensor, manifold=Euclidean(), man_dim=y.man_dim)

    def expmap(self, x: ManifoldTensor, v: ManifoldTensor) -> ManifoldTensor:
        dim = check_dims_with_broadcasting(x, v)
        new_tensor = expmap(x=x.tensor, v=v.tensor, c=self.c, dim=dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=dim)

    def logmap(self, x: ManifoldTensor, y: ManifoldTensor) -> ManifoldTensor:
        dim = check_dims_with_broadcasting(x, y)
        new_tensor = logmap(x=x.tensor, y=y.tensor, c=self.c, dim=dim)
        return ManifoldTensor(data=new_tensor, manifold=Euclidean(), man_dim=dim)

    def gyration(self, u: ManifoldTensor, v: ManifoldTensor, w: ManifoldTensor) -> ManifoldTensor:
        dim = check_dims_with_broadcasting(u, v, w)
        new_tensor = gyration(u=u.tensor, v=v.tensor, w=w.tensor, c=self.c, dim=dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=dim)

    def transp(self, x: ManifoldTensor, y: ManifoldTensor, v: ManifoldTensor) -> ManifoldTensor:
        dim = check_dims_with_broadcasting(x, y, v)
        new_tensor = transp(x=x.tensor, y=y.tensor, v=v.tensor, c=self.c, dim=dim)
        return ManifoldTensor(data=new_tensor, manifold=Euclidean(), man_dim=dim)

    def dist(self, x: ManifoldTensor, y: ManifoldTensor) -> Tensor:
        dim = check_dims_with_broadcasting(x, y)
        return dist(x=x.tensor, y=y.tensor, c=self.c, dim=dim)

    def inner(
        self, x: ManifoldTensor, u: ManifoldTensor, v: ManifoldTensor, keepdim: bool = False
    ) -> Tensor:
        dim = check_dims_with_broadcasting(x, u, v)
        return inner(x=x.tensor, u=u.tensor, v=v.tensor, c=self.c, dim=dim, keepdim=keepdim)

    def euc_to_tangent(self, x: ManifoldTensor, u: ManifoldTensor) -> ManifoldTensor:
        dim = check_dims_with_broadcasting(x, u)
        new_tensor = euc_to_tangent(x=x.tensor, u=u.tensor, c=self.c, dim=x.man_dim)
        return ManifoldTensor(data=new_tensor, manifold=Euclidean(), man_dim=dim)

    def hyperplane_dists(self, x: ManifoldTensor, z: ManifoldTensor, r: Optional[Tensor]) -> Tensor:
        # TODO: check dimensions
        return poincare_hyperplane_dists(x=x.tensor, z=z.tensor, r=r, c=self.c)

    def fully_connected(
        self, x: ManifoldTensor, z: ManifoldTensor, bias: Optional[Tensor]
    ) -> ManifoldTensor:
        # TODO: check dimensions
        new_tensor = poincare_fully_connected(x=x.tensor, z=z.tensor, bias=bias, c=self.c)
        new_tensor = ManifoldTensor(data=new_tensor, manifold=self, man_dim=-1)
        return self.project(new_tensor)

    def frechet_mean(self, x: ManifoldTensor, w: Optional[Tensor] = None) -> ManifoldTensor:
        # TODO: make frechet mean have dim options and add dim checks
        new_tensor = frechet_mean(x=x.tensor, c=self.c, w=w)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=-1)

    def frechet_variance(
        self, x: ManifoldTensor, mu: ManifoldTensor, dim: int = -1, w: Optional[Tensor] = None
    ) -> Tensor:
        # TODO: make frechet variance have proper dim options and add dim checks
        return frechet_variance(x=x.tensor, mu=mu.tensor, c=self.c, dim=dim, w=w)

    def construct_dl_parameters(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> tuple[ManifoldParameter, Optional[nn.Parameter]]:
        weight = ManifoldParameter(
            data=empty(in_features, out_features),
            manifold=Euclidean(),
            man_dim=-1,
        )

        if bias:
            b = nn.Parameter(data=empty(out_features))
        else:
            b = None

        return weight, b

    def reset_parameters(self, weight: ManifoldParameter, bias: Optional[nn.Parameter]) -> None:
        in_features, out_features = weight.size()
        if in_features <= out_features:
            with no_grad():
                weight.tensor.copy_(1 / 2 * eye(in_features, out_features))
        else:
            normal_(
                weight.tensor,
                mean=0,
                std=(2 * in_features * out_features) ** -0.5,
            )
        if bias is not None:
            zeros_(bias)

    def unfold(
        self,
        input: ManifoldTensor,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 1,
        padding: _size_2_t = 0,
        stride: _size_2_t = 1,
    ):
        # TODO: doesn't work with tuple dilation, padding, and stride. Should add this.
        # TODO: may have to cache some of this stuff for efficiency.
        in_channels = input.size(1)
        if len(kernel_size) == 2:
            kernel_vol = kernel_size[0] * kernel_size[1]
        else:
            kernel_vol = kernel_size**2
            kernel_size = (kernel_size, kernel_size)

        beta_ni = beta_func(in_channels / 2, 1 / 2)
        beta_n = beta_func(in_channels * kernel_vol / 2, 1 / 2)

        input = self.logmap0(input)
        input.tensor = input.tensor * beta_n / beta_ni
        new_tensor = unfold(
            input=input.tensor,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        new_tensor = new_tensor.transpose(1, 2)
        new_tensor = ManifoldTensor(data=new_tensor, manifold=self, man_dim=2)
        return self.expmap0(new_tensor)
