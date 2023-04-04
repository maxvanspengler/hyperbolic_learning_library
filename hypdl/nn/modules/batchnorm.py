from torch import Tensor, tensor, zeros
from torch.nn import Module, Parameter

from ...manifolds import Manifold


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
        use_midpoint: bool = True,
    ) -> None:
        super(HBatchNorm, self).__init__()
        self.features = features
        self.manifold = manifold
        self.use_midpoint = use_midpoint

        self.bias = Parameter(zeros(features))
        self.weight = Parameter(tensor(1.0))

    def forward(self, x):
        bias_on_manifold = self.manifold.expmap0(self.bias, dim=-1)

        if self.use_midpoint:
            # TODO: add midpoint to manifolds
            # input_mean = self.manifold.midpoint(x, self.ball.c, vec_dim=-1, batch_dim=0)
            pass
        else:
            input_mean = self.manifold.frechet_mean(x=x)

        input_var = self.manifold.frechet_variance(x, input_mean, dim=-1)

        input_logm = self.manifold.transp(
            x=input_mean,
            y=bias_on_manifold,
            v=self.manifold.logmap(input_mean, x),
        )

        input_logm = (self.weight / (input_var + 1e-6)).sqrt() * input_logm

        output = self.manifold.expmap(bias_on_manifold.unsqueeze(-2), input_logm)

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
        use_midpoint: bool = True,
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

    def forward(self, x: Tensor) -> Tensor:
        batch_size, height, width = x.size(0), x.size(2), x.size(3)
        x = x.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)
        x = self.norm(x)
        x = x.reshape(batch_size, height, width, self.features).permute(0, 3, 1, 2)
        return x
