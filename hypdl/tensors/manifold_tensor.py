from typing import Any, Optional

from torch import Tensor, tensor

from hypdl.manifolds import Manifold


class ManifoldTensor(object):
    def __init__(
        self, data, manifold: Manifold, man_dim: int = -1, requires_grad: bool = False
    ) -> None:
        if isinstance(data, Tensor):
            self.tensor = data
        else:
            self.tensor = tensor(data, requires_grad=requires_grad)

        self.manifold = manifold

        if man_dim > 0:
            self.man_dim = man_dim
        else:
            self.man_dim = self.tensor.dim() + man_dim
            if self.man_dim < 0:
                raise ValueError(
                    f"Dimension out of range (expected to be in range of "
                    f"{[-self.tensor.dim() - 1, self.tensor.dim()]}, but got {man_dim})"
                )

    def __getattr__(self, name: str) -> Any:
        # TODO: go through https://pytorch.org/docs/stable/tensors.html and check which methods
        # are relevant.
        if hasattr(self.tensor, name):
            torch_attribute = getattr(self.tensor, name)

            if callable(torch_attribute):
                raise AttributeError(
                    f"Attempting to apply the torch.Tensor method {name} on a ManifoldTensor."
                    f"Use ManifoldTensor.tensor.{name} instead."
                )
            else:
                return torch_attribute

        else:
            raise AttributeError(
                f"Neither {self.__class__.__name__}, nor torch.Tensor has attribute {name}"
            )

    def project(self):
        return self.manifold.project(x=self, dim=self.man_dim)

    def cuda(self):
        new_tensor = self.tensor.cuda()
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def cpu(self):
        new_tensor = self.tensor.cpu()
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def to(self, *args, **kwargs):
        new_tensor = self.tensor.to(*args, **kwargs)
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.tensor.size()
        else:
            return self.tensor.size(dim)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # TODO: check if there are torch functions that should be allowed
        raise TypeError(
            f"Attempting to apply the torch function {func} on a ManifoldTensor."
            f"Use ManifoldTensor.tensor as argument to {func} instead."
        )
