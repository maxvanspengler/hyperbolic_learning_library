from typing import Any

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
        if hasattr(self.tensor, name):
            torch_attribute = getattr(self.tensor, name)

            if callable(torch_attribute):

                def wrapped_torch_method(*args, **kwargs):
                    return_val = torch_attribute(*args, **kwargs)
                    if isinstance(return_val, Tensor):
                        new_tensor = ManifoldTensor(
                            data=return_val, manifold=self.manifold, man_dim=self.man_dim
                        )
                        return new_tensor
                    else:
                        return return_val

                return wrapped_torch_method

            else:
                return torch_attribute

        else:
            raise AttributeError(
                f"Neither {self.__class__.__name__}, nor torch.Tensor has attribute {name}"
            )

    def __add__(self, other: Any):
        if isinstance(other, ManifoldTensor):
            new_tensor = self.tensor + other.tensor
        else:
            new_tensor = self.tensor + other
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def __radd__(self, other: Any):
        if isinstance(other, ManifoldTensor):
            new_tensor = self.tensor + other.tensor
        else:
            new_tensor = self.tensor + other
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def __sub__(self, other: Any):
        if isinstance(other, ManifoldTensor):
            new_tensor = self.tensor - other.tensor
        else:
            new_tensor = self.tensor - other
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def __rsub__(self, other: Any):
        if isinstance(other, ManifoldTensor):
            new_tensor = other.tensor - self.tensor
        else:
            new_tensor = other - self.tensor
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def __mul__(self, other: Any):
        if isinstance(other, ManifoldTensor):
            new_tensor = self.tensor * other.tensor
        else:
            new_tensor = self.tensor * other
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def __rmul__(self, other: Any):
        if isinstance(other, ManifoldTensor):
            new_tensor = self.tensor * other.tensor
        else:
            new_tensor = self.tensor * other
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def __truediv__(self, other: Any):
        if isinstance(other, ManifoldTensor):
            new_tensor = self.tensor / other.tensor
        else:
            new_tensor = self.tensor / other
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def __rtruediv__(self, other: Any):
        if isinstance(other, ManifoldTensor):
            new_tensor = self.tensor / other.tensor
        else:
            new_tensor = self.tensor / other
        return ManifoldTensor(data=new_tensor, manifold=self.manifold, man_dim=self.man_dim)

    def project(self):
        return self.manifold.project(x=self, dim=self.man_dim)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        for arg in args:
            if isinstance(arg, cls):
                manifold = arg.manifold
                man_dim = arg.man_dim
                break
        args = [arg.tensor if hasattr(arg, "tensor") else arg for arg in args]
        ret = func(*args, **kwargs)
        return ManifoldTensor(ret, manifold=manifold, man_dim=man_dim)
