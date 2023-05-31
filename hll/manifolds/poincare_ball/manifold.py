from typing import Optional, Union

from torch import Tensor, empty, eye, no_grad
from torch.nn import Parameter
from torch.nn.common_types import _size_2_t
from torch.nn.functional import softplus, unfold
from torch.nn.init import normal_, zeros_

from hll.manifolds.base import Manifold
from hll.manifolds.euclidean import Euclidean
from hll.manifolds.poincare_ball.curvature import Curvature
from hll.tensors import ManifoldParameter, ManifoldTensor, TangentTensor
from hll.utils.math import beta_func
from hll.utils.tensor_utils import (
    check_dims_with_broadcasting,
    check_tangent_tensor_positions,
)

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
from .math.stats import frechet_mean, frechet_variance, midpoint


class PoincareBall(Manifold):
    """Class representing the Poincare ball model of hyperbolic space.

    Implementation based on the geoopt implementation, but changed to use
    hyperbolic torch functions.

    Attributes:
        curvature:
            Curvature of the manifold.
        c:
            Scalar tensor representing curvature returned by curvature.forward().

    """

    def __init__(self, curvature: Curvature):
        """Initializes an instance of PoincareBall manifold.

        Args:
            curvature:
                Curvature of the manifold.

        Examples:
            >>> from hypdl.manifolds import PoincareBall, Curvature
            >>> curvature = Curvature(value=1.0)
            >>> manifold = Manifold(curvature=curvature)

        """
        super(PoincareBall, self).__init__()
        self.curvature = curvature

    @property
    def c() -> Tensor:
        return self.curvature()

    def mobius_add(self, x: ManifoldTensor, y: ManifoldTensor) -> ManifoldTensor:
        dim = check_dims_with_broadcasting(x, y)
        new_tensor = mobius_add(x=x, y=y, c=self.c, dim=dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=dim)

    def project(self, x: ManifoldTensor, eps: float = -1.0) -> ManifoldTensor:
        new_tensor = project(x=x.tensor, c=self.c, dim=x.man_dim, eps=eps)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=x.man_dim)

    def expmap(self, v: TangentTensor) -> ManifoldTensor:
        dim = v.broadcasted_man_dim
        if v.manifold_points is None:
            new_tensor = expmap0(v=v.tensor, c=self.c, dim=dim)
        else:
            new_tensor = expmap(x=v.manifold_points.tensor, v=v.tensor, c=self.c, dim=dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=dim)

    def logmap(self, x: Optional[ManifoldTensor], y: ManifoldTensor):
        if x is None:
            dim = y.man_dim
            new_tensor = logmap0(y=y.tensor, c=self.c, dim=y.man_dim)
        else:
            dim = check_dims_with_broadcasting(x, y)
            new_tensor = logmap(x=x.tensor, y=y.tensor, c=self.c, dim=dim)
        return TangentTensor(data=new_tensor, manifold_points=x, manifold=self, man_dim=dim)

    def gyration(self, u: ManifoldTensor, v: ManifoldTensor, w: ManifoldTensor) -> ManifoldTensor:
        dim = check_dims_with_broadcasting(u, v, w)
        new_tensor = gyration(u=u.tensor, v=v.tensor, w=w.tensor, c=self.c, dim=dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=dim)

    def transp(self, v: TangentTensor, y: ManifoldTensor) -> TangentTensor:
        dim = check_dims_with_broadcasting(v, y)
        tangent_vectors = transp(
            x=v.manifold_points.tensor, y=y.tensor, v=v.tensor, c=self.c, dim=dim
        )
        return TangentTensor(
            data=tangent_vectors,
            manifold_points=y,
            manifold=self,
            man_dim=dim,
        )

    def dist(self, x: ManifoldTensor, y: ManifoldTensor) -> Tensor:
        dim = check_dims_with_broadcasting(x, y)
        return dist(x=x.tensor, y=y.tensor, c=self.c, dim=dim)

    def inner(
        self, u: TangentTensor, v: TangentTensor, keepdim: bool = False, safe_mode: bool = True
    ) -> Tensor:
        dim = check_dims_with_broadcasting(u, v)
        if safe_mode:
            check_tangent_tensor_positions(u, v)

        return inner(
            x=u.manifold_points.tensor, u=u.tensor, v=v.tensor, c=self.c, dim=dim, keepdim=keepdim
        )

    def euc_to_tangent(self, x: ManifoldTensor, u: ManifoldTensor) -> TangentTensor:
        dim = check_dims_with_broadcasting(x, u)
        tangent_vectors = euc_to_tangent(x=x.tensor, u=u.tensor, c=self.c, dim=x.man_dim)
        return TangentTensor(
            data=tangent_vectors,
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
        return poincare_hyperplane_dists(x=x.tensor, z=z.tensor, r=r, c=self.c)

    def fully_connected(
        self, x: ManifoldTensor, z: ManifoldTensor, bias: Optional[Tensor]
    ) -> ManifoldTensor:
        if x.man_dim != 1 or z.man_dim != 0:
            raise ValueError(
                f"Expected the manifold dimension of the inputs to be 1 and the manifold "
                f"dimension of the hyperplane orientations to be 0, but got {x.man_dim} and "
                f"{z.man_dim}, respectively"
            )
        new_tensor = poincare_fully_connected(x=x.tensor, z=z.tensor, bias=bias, c=self.c)
        new_tensor = ManifoldTensor(data=new_tensor, manifold=self, man_dim=-1)
        return self.project(new_tensor)

    def frechet_mean(
        self,
        x: ManifoldTensor,
        batch_dim: Union[int, list[int]] = 0,
        keepdim: bool = False,
    ) -> ManifoldTensor:
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]
        output_man_dim = x.man_dim - sum(bd < x.man_dim for bd in batch_dim)
        new_tensor = frechet_mean(
            x=x.tensor, c=self.c, vec_dim=x.man_dim, batch_dim=batch_dim, keepdim=keepdim
        )
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=output_man_dim)

    def midpoint(
        self,
        x: ManifoldTensor,
        batch_dim: int = 0,
        w: Optional[Tensor] = None,
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

        new_tensor = midpoint(
            x=x.tensor, c=self.c, man_dim=x.man_dim, batch_dim=batch_dim, w=w, keepdim=keepdim
        )
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=new_man_dim)

    def frechet_variance(
        self,
        x: ManifoldTensor,
        mu: Optional[ManifoldTensor] = None,
        batch_dim: Union[int, list[int]] = -1,
        keepdim: bool = False,
    ) -> Tensor:
        if mu is not None:
            mu = mu.tensor

        # TODO: Check if x and mu have compatible man_dims
        return frechet_variance(
            x=x.tensor,
            c=self.c,
            mu=mu,
            vec_dim=x.man_dim,
            batch_dim=batch_dim,
            keepdim=keepdim,
        )

    def construct_dl_parameters(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> tuple[ManifoldParameter, Optional[Parameter]]:
        weight = ManifoldParameter(
            data=empty(in_features, out_features),
            manifold=Euclidean(),
            man_dim=0,
        )

        if bias:
            b = Parameter(data=empty(out_features))
        else:
            b = None

        return weight, b

    def reset_parameters(self, weight: ManifoldParameter, bias: Optional[Parameter]) -> None:
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
    ) -> ManifoldTensor:
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

        input = self.logmap(x=None, y=input)
        input.tensor = input.tensor * beta_n / beta_ni
        new_tensor = unfold(
            input=input.tensor,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        new_tensor = new_tensor.transpose(1, 2)
        new_tensor = TangentTensor(data=new_tensor, manifold_points=None, manifold=self, man_dim=2)
        return self.expmap(new_tensor)
