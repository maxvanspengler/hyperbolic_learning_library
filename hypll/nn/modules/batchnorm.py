from torch import tensor, zeros
from torch.nn import Module, Parameter

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor, TangentTensor
from hypll.utils.layer_utils import check_if_manifolds_match


class HBatchNorm(Module):
    """
    Basic implementation of hyperbolic batch normalization.

    Based on:
        https://arxiv.org/abs/2003.00335
    """

    def __init__(
        self,
        features: int,
        manifold: Manifold,
        use_midpoint: bool = False,
    ) -> None:
        super(HBatchNorm, self).__init__()
        self.features = features
        self.manifold = manifold
        self.use_midpoint = use_midpoint

        # TODO: Store bias on manifold
        self.bias = Parameter(zeros(features))
        self.weight = Parameter(tensor(1.0))

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=x)
        bias_on_manifold = self.manifold.expmap(
            v=TangentTensor(data=self.bias, manifold_points=None, manifold=self.manifold)
        )

        if self.use_midpoint:
            input_mean = self.manifold.midpoint(x=x, batch_dim=0)
        else:
            input_mean = self.manifold.frechet_mean(x=x, batch_dim=0)

        input_var = self.manifold.frechet_variance(x=x, mu=input_mean, batch_dim=0)

        input_logm = self.manifold.transp(
            v=self.manifold.logmap(input_mean, x),
            y=bias_on_manifold,
        )

        input_logm.tensor = (self.weight / (input_var + 1e-6)).sqrt() * input_logm.tensor

        output = self.manifold.expmap(input_logm)

        return output


class HBatchNorm2d(Module):
    """
    2D implementation of hyperbolic batch normalization.

    Based on:
        https://arxiv.org/abs/2003.00335
    """

    def __init__(
        self,
        features: int,
        manifold: Manifold,
        use_midpoint: bool = False,
    ) -> None:
        super(HBatchNorm2d, self).__init__()
        self.features = features
        self.manifold = manifold
        self.use_midpoint = use_midpoint

        self.norm = HBatchNorm(
            features=features,
            manifold=manifold,
            use_midpoint=use_midpoint,
        )

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=x)
        batch_size, height, width = x.size(0), x.size(2), x.size(3)
        flat_x = ManifoldTensor(
            data=x.tensor.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2),
            manifold=x.manifold,
            man_dim=-1,
        )
        flat_x = self.norm(flat_x)
        new_tensor = flat_x.tensor.reshape(batch_size, height, width, self.features).permute(
            0, 3, 1, 2
        )
        return ManifoldTensor(data=new_tensor, manifold=x.manifold, man_dim=1)
