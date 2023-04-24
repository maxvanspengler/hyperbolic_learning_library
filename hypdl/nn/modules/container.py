from torch.nn import Module, Sequential

from hypdl.manifolds import Manifold
from hypdl.tensors import ManifoldTensor
from hypdl.utils.layer_utils import check_if_manifolds_match


class TangentSequential(Module):
    def __init__(self, seq: Sequential, manifold: Manifold) -> None:
        super(TangentSequential, self).__init__()
        self.seq = seq
        self.manifold = manifold

    def forward(self, input: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=input)
        man_dim = input.man_dim

        input = self.manifold.logmap0(input, dim=man_dim)
        for module in self.seq:
            input = module(input)
        return self.manifold.expmap0(input, dim=man_dim)
