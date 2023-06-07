from typing import Any

import torch
from torch import Tensor, tensor
from torch.nn import Parameter

from hypll.manifolds import Manifold
from hypll.tensors.manifold_tensor import ManifoldTensor


class ManifoldParameter(ManifoldTensor, Parameter):
    _allowed_methods = [
        torch._has_compatible_shallow_copy_type,  # Required for torch.nn.Parameter
        torch.Tensor.copy_,  # Required to load ManifoldParameters state dicts
    ]

    def __new__(cls, data, manifold, man_dim, requires_grad=True):
        return super(ManifoldTensor, cls).__new__(cls)

    # TODO: Create a mixin class containing the methods for this class and for ManifoldTensor
    # to avoid all the boilerplate stuff.
    def __init__(
        self, data, manifold: Manifold, man_dim: int = -1, requires_grad: bool = True
    ) -> None:
        super(ManifoldParameter, self).__init__(data=data, manifold=manifold)
        if isinstance(data, Parameter):
            self.tensor = data
        elif isinstance(data, Tensor):
            self.tensor = Parameter(data=data, requires_grad=requires_grad)
        else:
            self.tensor = Parameter(data=tensor(data), requires_grad=requires_grad)

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

    def __getattr__(self, name: str) -> Any:
        # TODO: go through https://pytorch.org/docs/stable/tensors.html and check which methods
        # are relevant.
        if hasattr(self.tensor, name):
            torch_attribute = getattr(self.tensor, name)

            if callable(torch_attribute):
                raise AttributeError(
                    f"Attempting to apply the torch.nn.Parameter method {name} on a ManifoldParameter."
                    f"Use ManifoldTensor.tensor.{name} instead."
                )
            else:
                return torch_attribute

        else:
            raise AttributeError(
                f"Neither {self.__class__.__name__}, nor torch.Tensor has attribute {name}"
            )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func.__class__.__name__ == "method-wrapper" or func in cls._allowed_methods:
            args = [a.tensor if isinstance(a, ManifoldTensor) else a for a in args]
            if kwargs is None:
                kwargs = {}
            kwargs = {k: (v.tensor if isinstance(v, ManifoldTensor) else v) for k, v in kwargs}
            return func(*args, **kwargs)
        # if func.__name__ == "__get__":
        #     return func(args[0].tensor)
        # TODO: check if there are torch functions that should be allowed
        raise TypeError(
            f"Attempting to apply the torch function {func} on a ManifoldParameter. "
            f"Use ManifoldParameter.tensor as argument to {func} instead."
        )
