from typing import Optional

import torch.nn as nn
from torch import Tensor, as_tensor, empty, eye, float32, no_grad
from torch.nn.common_types import _size_2_t
from torch.nn.functional import unfold
from torch.nn.init import normal_, zeros_

from hypdl.manifolds.base import Manifold
from hypdl.tensors import ManifoldParameter, ManifoldTensor
from hypdl.utils.math import beta_func

from .math.diffgeom import (
    dist,
    euc_to_tangent,
    expmap,
    expmap0,
    gyration,
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
        return nn.functional.softplus(self.isp_c)

    def mobius_add(self, x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
        return mobius_add(x=x, y=y, c=self.c, dim=dim)

    def project(self, x: Tensor, dim: int = -1, eps: float = -1.0) -> Tensor:
        return project(x=x, c=self.c, dim=dim, eps=eps)

    def expmap0(self, v: Tensor, dim: int = -1) -> Tensor:
        return expmap0(v=v, c=self.c, dim=dim)

    def logmap0(self, y: Tensor, dim: int = -1) -> Tensor:
        return logmap0(y=y, c=self.c, dim=dim)

    def expmap(self, x: Tensor, v: Tensor, dim: int = -1) -> Tensor:
        return expmap(x=x, v=v, c=self.c, dim=dim)

    def logmap(self, x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
        return logmap(x=x, y=y, c=self.c, dim=dim)

    def gyration(self, u: Tensor, v: Tensor, w: Tensor, dim: int = -1) -> Tensor:
        return gyration(u=u, v=v, w=w, c=self.c, dim=dim)

    def transp(self, x: Tensor, y: Tensor, v: Tensor, dim: int = -1) -> Tensor:
        return transp(x=x, y=y, v=v, c=self.c, dim=dim)

    def dist(self, x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
        return dist(x=x, y=y, c=self.c, dim=dim)

    def euc_to_tangent(self, x: Tensor, u: Tensor, dim: int = -1) -> Tensor:
        return euc_to_tangent(x=x, u=u, c=self.c, dim=dim)

    def hyperplane_dists(self, x: Tensor, z: Tensor, r: Optional[Tensor]) -> Tensor:
        return poincare_hyperplane_dists(x=x, z=z, r=r, c=self.c)

    def fully_connected(self, x: Tensor, z: Tensor, bias: Optional[Tensor]) -> Tensor:
        y = poincare_fully_connected(x=x, z=z, bias=bias, c=self.c)
        return self.project(y, dim=-1)

    def frechet_mean(self, x: Tensor, w: Optional[Tensor] = None) -> Tensor:
        return frechet_mean(x=x, c=self.c, w=w)

    def frechet_variance(
        self, x: Tensor, mu: Tensor, dim: int = -1, w: Optional[Tensor] = None
    ) -> Tensor:
        return frechet_variance(x=x, mu=mu, c=self.c, dim=dim, w=w)

    def construct_dl_parameters(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> tuple[ManifoldParameter, Optional[ManifoldParameter]]:
        weight = ManifoldParameter(
            data=empty(in_features, out_features),
            manifold=self,
            man_dim=-1,
        )

        if bias:
            b = ManifoldParameter(
                data=empty(out_features),
                manifold=self,
                man_dim=-1,
            )
        else:
            b = None

        return weight, b

    def reset_parameters(
        self, weight: ManifoldParameter, bias: Optional[ManifoldParameter]
    ) -> None:
        in_features, out_features = weight.size()
        if in_features <= out_features:
            with no_grad():
                weight.copy_(1 / 2 * eye(in_features, out_features))
        else:
            normal_(
                weight,
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

        input = self.logmap0(input, dim=1)
        input = input * beta_n / beta_ni
        return unfold(
            input=input,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
