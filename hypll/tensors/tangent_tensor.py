from typing import Any, Optional

from torch import Tensor, broadcast_shapes, tensor

from hypll.manifolds import Manifold
from hypll.tensors.manifold_tensor import ManifoldTensor


class TangentTensor:
    def __init__(
        self,
        data,
        manifold_points: Optional[ManifoldTensor] = None,
        manifold: Optional[Manifold] = None,
        man_dim: int = -1,
        requires_grad: bool = False,
    ) -> None:
        # Create tangent vector tensor with correct device and dtype
        if isinstance(data, Tensor):
            self.tensor = data
        else:
            self.tensor = tensor(data, requires_grad=requires_grad)

        # Store manifold dimension as a nonnegative integer
        if man_dim >= 0:
            self.man_dim = man_dim
        else:
            self.man_dim = self.tensor.dim() + man_dim
            if self.man_dim < 0:
                raise ValueError(
                    f"Dimension out of range (expected to be in range of "
                    f"{[-self.tensor.dim() - 1, self.tensor.dim()]}, but got {man_dim})"
                )

        if manifold_points is not None:
            # Check if the manifold points and tangent vectors are broadcastable together
            try:
                broadcasted_size = broadcast_shapes(self.tensor.size(), manifold_points.size())
            except RuntimeError:
                raise ValueError(
                    f"The shapes of the manifold points tensor {manifold_points.size()} and "
                    f"the tangent vector tensor {self.tensor.size()} are not broadcastable "
                    f"togther."
                )

            # Check if the manifold dimensions match after broadcasting
            dim = len(broadcasted_size)
            broadcast_man_dims = [
                manifold_points.man_dim + dim - manifold_points.tensor.dim(),
                self.man_dim + dim - self.tensor.dim(),
            ]
            if broadcast_man_dims[0] != broadcast_man_dims[1]:
                raise ValueError(
                    f"After broadcasting the manifold points with the tangent vectors, the "
                    f"manifold dimension computed from the manifold points should match the "
                    f"manifold dimension computed from the supplied man_dim, but these are"
                    f"{broadcast_man_dims}, respectively."
                )

        # Check if the supplied manifolds match
        if manifold_points is not None and manifold is not None:
            if manifold_points.manifold != manifold:
                raise ValueError(
                    f"The manifold of the manifold_points and the provided manifold should match, "
                    f"but are {manifold_points.manifold} and {manifold}, respectively."
                )

        self.manifold_points = manifold_points
        self.manifold = manifold or manifold_points.manifold

    def __getattr__(self, name: str) -> Any:
        # TODO: go through https://pytorch.org/docs/stable/tensors.html and check which methods
        # are relevant.
        if hasattr(self.tensor, name):
            torch_attribute = getattr(self.tensor, name)

            if callable(torch_attribute):
                raise AttributeError(
                    f"Attempting to apply the torch.Tensor method {name} on a TangentTensor."
                    f"Use TangentTensor.tensor.{name} or TangentTensor.manifold_points.tensor "
                    f"instead."
                )
            else:
                return torch_attribute

        else:
            raise AttributeError(
                f"Neither {self.__class__.__name__}, nor torch.Tensor has attribute {name}"
            )

    def __hash__(self):
        """Returns the Python unique identifier of the object.

        Note: This is how PyTorch implements hash of tensors. See also:
        https://github.com/pytorch/pytorch/issues/2569.

        """
        return id(self)

    def cuda(self, device=None):
        new_tensor = self.tensor.cuda(device)
        new_manifold_points = self.manifold_points.cuda(device)
        return TangentTensor(
            data=new_tensor,
            manifold_points=new_manifold_points,
            manifold=self.manifold,
            man_dim=self.man_dim,
        )

    def cpu(self):
        new_tensor = self.tensor.cpu()
        new_manifold_points = self.manifold_points.cpu()
        return TangentTensor(
            data=new_tensor,
            manifold_points=new_manifold_points,
            manifold=self.manifold,
            man_dim=self.man_dim,
        )

    def to(self, *args, **kwargs):
        new_tensor = self.tensor.to(*args, **kwargs)
        new_manifold_points = self.manifold_points(*args, **kwargs)
        return TangentTensor(
            data=new_tensor,
            manifold_points=new_manifold_points,
            manifold=self.manifold,
            man_dim=self.man_dim,
        )

    def size(self, dim: Optional[int] = None):
        if self.manifold_points is None:
            manifold_points_size = None
        manifold_points_size = (
            self.manifold_points.size() if self.manifold_points is not None else ()
        )
        broadcasted_size = broadcast_shapes(self.tensor.size(), manifold_points_size)
        if dim is None:
            return broadcasted_size
        else:
            return broadcasted_size[dim]

    @property
    def broadcasted_man_dim(self):
        return self.man_dim + self.dim() - self.tensor.dim()

    def dim(self):
        return len(self.size())

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # TODO: check if there are torch functions that should be allowed
        raise TypeError(
            f"Attempting to apply the torch function {func} on a TangentTensor."
            f"Use TangentTensor.tensor as argument to {func} instead."
        )
