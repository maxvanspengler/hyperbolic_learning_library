from typing import Any

from torch import Tensor, tensor
from torch.nn import Module, Parameter

from hypdl.manifolds import Manifold
from hypdl.tensors.manifold_tensor import ManifoldTensor


class ManifoldParameter(Module):
    # TODO: Create a mixin class containing the methods for this class and for ManifoldTensor
    # to avoid all the boilerplate stuff.
    def __init__(
        self, data, manifold: Manifold, man_dim: int = -1, requires_grad: bool = True
    ) -> None:
        super(ManifoldParameter, self).__init__()
        if isinstance(data, Parameter):
            self.tensor = data
        elif isinstance(data, Tensor):
            self.tensor = Parameter(data=data, requires_grad=requires_grad)
        else:
            self.tensor = Parameter(data=tensor(data), requires_grad=requires_grad)

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
        if name == "tensor":
            if name in self._parameters:
                return self._parameters[name]
            else:
                raise AttributeError(f"ManifoldParameter has no registered parameter attribute")

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

        raise AttributeError(
            f"Neither {self.__class__.__name__}, nor torch.Tensor has attribute {name}"
        )

    def __add__(self, other: Any):
        return ManifoldTensor.__add__(self=self, other=other)

    def __radd__(self, other: Any):
        return ManifoldTensor.__radd__(self=self, other=other)

    def __sub__(self, other: Any):
        return ManifoldTensor.__sub__(self=self, other=other)

    def __rsub__(self, other: Any):
        return ManifoldTensor.__rsub__(self=self, other=other)

    def __mul__(self, other: Any):
        return ManifoldTensor.__mul__(self=self, other=other)

    def __rmul__(self, other: Any):
        return ManifoldTensor.__rmul__(self=self, other=other)

    def __truediv__(self, other: Any):
        return ManifoldTensor.__truediv__(self=self, other=other)

    def __rtruediv__(self, other: Any):
        return ManifoldTensor.__rtruediv__(self=self, other=other)

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
