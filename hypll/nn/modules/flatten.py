from torch.nn import Module, functional

from hypll.tensors import ManifoldTensor


class HFlatten(Module):
    """Flattens a contiguous range of dims into a tensor.

    Attributes:
        start_dim:
            First dimension to flatten (default = 1).
        end_dim:
            Last dimension to flatten (default = -1).

    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super(HFlatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        """Flattens the manifold input tensor."""
        return x.flatten(start_dim=self.start_dim, end_dim=self.end_dim)
