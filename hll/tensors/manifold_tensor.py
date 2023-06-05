from __future__ import annotations

from typing import Optional

from torch import Tensor, long, tensor

from hll.manifolds import Manifold


class ManifoldTensor:
    """Represents a tensor on a manifold.

    Attributes:
        tensor:
            Torch tensor of points on the manifold.
        manifold:
            Manifold instance.
        man_dim:
            Dimension along which points are on the manifold.

    """

    def __init__(
        self, data: Tensor, manifold: Manifold, man_dim: int = -1, requires_grad: bool = False
    ) -> None:
        """Creates an instance of ManifoldTensor.

        Args:
            data:
                Torch tensor of points on the manifold.
            manifold:
                Manifold instance.
            man_dim:
                Dimension along which points are on the manifold. -1 by default.

        TODO(Philipp, 05/23): Let's get rid of requires_grad if possible.

        """
        self.tensor = data if isinstance(data, Tensor) else tensor(data, requires_grad=True)
        self.manifold = manifold

        if man_dim >= 0:
            self.man_dim = man_dim
        else:
            self.man_dim = self.tensor.dim() + man_dim
            if self.man_dim < 0:
                raise ValueError(
                    f"Dimension out of range (expected to be in range of "
                    f"{[-self.tensor.dim() - 1, self.tensor.dim()]}, but got {man_dim})"
                )

    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], Tensor) and args[0].dtype == long:
            new_tensor = self.tensor.__getitem__(*args)
            return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=-1)

        arg_list = list(args[0])
        if Ellipsis in arg_list:
            ell_id = arg_list.index(Ellipsis)
            colon_repeats = self.dim() - sum(1 for a in arg_list if a is not None) + 1
            arg_list[ell_id : ell_id + 1] = colon_repeats * [slice(None, None, None)]

        new_tensor = self.tensor.__getitem__(*args)
        output_man_dim = self.man_dim
        counter = self.man_dim + 1

        for arg in arg_list:
            if arg is None:
                output_man_dim += 1
                continue
            elif isinstance(arg, int):
                output_man_dim -= 1
                counter -= 1
            else:
                counter -= 1

            if counter == 0:
                if isinstance(arg, int) or isinstance(arg, list):
                    raise ValueError(
                        f"Attempting to slice into the manifold dimension, but this is not a "
                        "valid operation"
                    )
                break

        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=output_man_dim)

    def cpu(self) -> ManifoldTensor:
        """Returns a copy of this object with self.tensor in CPU memory."""
        new_tensor = self.tensor.cpu()
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def cuda(self, device=None) -> ManifoldTensor:
        """Returns a copy of this object with self.tensor in CUDA memory."""
        new_tensor = self.tensor.cuda(device)
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def dim(self) -> int:
        """Returns the number of dimensions of self.tensor."""
        return self.tensor.dim()

    def detach(self) -> ManifoldTensor:
        """Returns a new Tensor, detached from the current graph."""
        detached = self.tensor.detach()
        return ManifoldTensor(data=detached, manifold=self.manifold, man_dim=self.man_dim)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> ManifoldTensor:
        """Flattens tensor by reshaping it. If start_dim or end_dim are passed,
        only dimensions starting with start_dim and ending with end_dim are flattend.

        """
        return self.manifold.flatten(self, start_dim=start_dim, end_dim=end_dim)

    @property
    def is_cpu(self):
        return self.tensor.is_cpu

    @property
    def is_cuda(self):
        return self.tensor.is_cuda

    def is_floating_point(self) -> bool:
        """Returns true if the tensor is of dtype float."""
        return self.tensor.is_floating_point()

    def project(self) -> ManifoldTensor:
        """Projects the tensor to the manifold."""
        return self.manifold.project(x=self)

    @property
    def shape(self):
        """Alias for size()."""
        return self.size()

    def size(self, dim: Optional[int] = None):
        """Returns the size of self.tensor."""
        if dim is None:
            return self.tensor.size()
        else:
            return self.tensor.size(dim)

    def to(self, *args, **kwargs) -> ManifoldTensor:
        """Returns a new tensor with the specified device and (optional) dtype."""
        new_tensor = self.tensor.to(*args, **kwargs)
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def transpose(self, dim0: int, dim1: int) -> ManifoldTensor:
        """Returns a transposed version of the manifold tensor. The given dimensions
        dim0 and dim1 are swapped.
        """
        if self.man_dim == dim0:
            new_man_dim = dim1
        elif self.man_dim == dim1:
            new_man_dim = dim0
        new_tensor = self.tensor.transpose(dim0, dim1)
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=new_man_dim)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # TODO: check if there are torch functions that should be allowed
        raise TypeError(
            f"Attempting to apply the torch function {func} on a ManifoldTensor. "
            f"Use ManifoldTensor.tensor as argument to {func} instead."
        )
