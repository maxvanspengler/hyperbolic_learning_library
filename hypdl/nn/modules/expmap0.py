from torch import Tensor
from torch.nn import Module

from hypdl.manifolds import Manifold
from hypdl.tensors import ManifoldTensor, TangentTensor


class ExpMap0(Module):
    """Applies the exponential map at the origin to input tensors.

    Attributes:
        manifold:
            Manifold instance.
        man_dim:
            Dimension along which points will be on the manifold.

    """

    def __init__(self, manifold: Manifold, man_dim: int):
        super(ExpMap0, self).__init__()
        self.manifold = manifold
        self.man_dim = man_dim

    def forward(self, x: Tensor) -> ManifoldTensor:
        """Applies the exponential map at the origin and returns a manifold tensor."""
        tangent_tensor = TangentTensor(data=x, manifold=self.manifold, man_dim=self.man_dim)
        return self.manifold.expmap(tangent_tensor)
