from typing import Optional

import torch.nn as nn
from torch import Tensor, as_tensor, float32

from ..base import Manifold
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
from .math.linalg import poincare_fully_connected, poincare_mlr
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

    def mlr(self, x: Tensor, z: Tensor, r: Tensor) -> Tensor:
        return poincare_mlr(x=x, z=z, r=r, c=self.c)

    def fully_connected(self, x: Tensor, z: Tensor, bias: Tensor) -> Tensor:
        y = poincare_fully_connected(x=x, z=z, bias=bias, c=self.c)
        return self.project(y, dim=-1)

    def frechet_mean(self, x: Tensor, w: Optional[Tensor] = None) -> Tensor:
        return frechet_mean(x=x, c=self.c, w=w)

    def frechet_variance(
        self, x: Tensor, mu: Tensor, dim: int = -1, w: Optional[Tensor] = None
    ) -> Tensor:
        return frechet_variance(x=x, mu=mu, c=self.c, dim=dim, w=w)
