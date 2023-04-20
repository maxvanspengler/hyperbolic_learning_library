from typing import Union

from torch import Tensor
from torch.nn import Parameter

from ..manifolds import Manifold
from .manifold_tensor import ManifoldTensor


class ManifoldParameter(ManifoldTensor, Parameter):
    def __new__(
        cls,
        data: Union[ManifoldTensor, Tensor, None],
        manifold: Manifold,
        man_dim: int = -1,
        requires_grad: bool = True,
    ):
        if data is None:
            data = ManifoldTensor(manifold=manifold)
        elif isinstance(data, ManifoldTensor) and manifold != data.manifold:
            raise ValueError(
                f"Manifold of input tensor is {data.manifold} while the input manifold is"
                f"{manifold}"
            )

        return ManifoldTensor._make_subclass(cls=cls, data=data, require_grad=requires_grad)

    def __init__(
        self,
        data: Union[ManifoldTensor, Tensor, None],
        manifold: Manifold,
        man_dim: int = -1,
        requires_grad: bool = True,
    ):
        if isinstance(data, ManifoldTensor):
            if man_dim != data.man_dim:
                raise ValueError(
                    f"Manifold dimension of input tensor is {data.man_dim} while the input manifold"
                    f"dimension is {man_dim}"
                )
            else:
                self.man_dim = man_dim
        else:
            if man_dim > 0:
                self.man_dim = man_dim
            else:
                self.man_dim = self.dim() + man_dim
                if self.man_dim < 0:
                    raise ValueError(
                        f"Dimension out of range (expected to be in range of {[-self.dim() - 1, self.dim()]}, but got {man_dim})"
                    )
