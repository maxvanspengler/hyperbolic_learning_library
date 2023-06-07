from torch import Tensor
from torch.nn import Module

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor, TangentTensor


class ChangeManifold(Module):
    """Changes the manifold of the input manifold tensor to the target manifold.

    Attributes:
        target_manifold:
            Manifold the output tensor should be on.

    """

    def __init__(self, target_manifold: Manifold):
        super(ChangeManifold, self).__init__()
        self.target_manifold = target_manifold

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        """Changes the manifold of the input tensor to self.target_manifold.

        By default, applies logmap and expmap at the origin to switch between
        manifold. Applies shortcuts if possible.

        Args:
            x:
                Input manifold tensor.

        Returns:
            Tensor on the target manifold.

        """

        match (x.manifold, self.target_manifold):
            # TODO(Philipp, 05/23): Apply shortcuts where possible: For example,
            # case (PoincareBall(), PoincareBall()): ...
            #
            # By default resort to logmap + expmap at the origin.
            case _:
                tangent_tensor = x.manifold.logmap(None, x)
                return self.target_manifold.expmap(tangent_tensor)
