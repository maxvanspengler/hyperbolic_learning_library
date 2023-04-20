from typing import Union

from torch import Tensor, tensor

from ..manifolds import Manifold


class ManifoldTensor(Tensor):
    def __new__(
        cls, *args, manifold: Manifold, man_dim: int = -1, requires_grad: bool = False, **kwargs
    ):
        device = kwargs["device"] if "device" in kwargs else "cpu"

        if isinstance(args[0], Tensor):
            data = args[0].to(device)
        else:
            data = tensor(*args, **kwargs)

        return Tensor._make_subclass(cls=cls, data=data, require_grad=requires_grad)

    def __init__(
        self, *args, manifold: Manifold, man_dim: int = -1, requires_grad: bool = False, **kwargs
    ) -> None:
        super(ManifoldTensor, self).__init__()
        self.manifold = manifold
        if man_dim > 0:
            self.man_dim = man_dim
        else:
            self.man_dim = self.dim() + man_dim
            if self.man_dim < 0:
                raise ValueError(
                    f"Dimension out of range (expected to be in range of {[-self.dim() - 1, self.dim()]}, but got {man_dim})"
                )

    def enorm(self, p: Union[int, str] = "fro", dim: int = -1, keepdim: bool = False):
        # TODO: This returns a ManifoldTensor, which doesn't make much sense. Change output to a
        # regular Tensor.
        return super(ManifoldTensor, self).norm(p=p, dim=dim, keepdim=keepdim)

    def project(self):
        return self.manifold.project(x=self, dim=self.man_dim)
