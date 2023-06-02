from typing import Optional

from torch import nn

from hll import nn as hnn
from hll.manifolds import PoincareBall
from hll.tensors import ManifoldTensor


class PoincareResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        manifold: PoincareBall,
        stride: int = 1,
        downsample: Optional[nn.Sequential] = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.manifold = manifold
        self.stride = stride
        self.downsample = downsample

        self.conv1 = hnn.HConvolution2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            manifold=manifold,
            stride=stride,
            padding=1,
        )
        self.bn1 = hnn.HBatchNorm2d(
            features=out_channels, manifold=manifold
        )
        self.relu = hnn.HReLU(manifold=self.manifold)
        self.conv2 = hnn.HConvolution2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            manifold=manifold,
            padding=1,
        )
        self.bn2 = hnn.HBatchNorm2d(
            features=out_channels, manifold=manifold
        )

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = self.manifold.mobius_add(x, residual) # TODO: add addition op to Manifold
        x = self.relu(x)

        return x


class PoincareResNet(nn.Module):
    def __init__(
        self,
        channel_sizes: list[int],
        group_depths: list[int],
        manifold: PoincareBall,
    ):
        self.channel_sizes = channel_sizes
        self.group_depths = group_depths
        self.manifold = manifold

        self.conv = hnn.HConvolution2d(
            in_channels=3,
            out_channels=channel_sizes[0],
            kernel_size=3,
            manifold=manifold,
            padding=1,
        )
        self.bn = hnn.HBatchNorm2d(
            features=channel_sizes[0], manifold=manifold
        )
        self.relu = hnn.HReLU(manifold=manifold)
        self.group1 = self._make_group(
            in_channels=channel_sizes[0],
            out_channels=channel_sizes[0],
            depth=group_depths[0],
        )
        self.group2 = self._make_group(
            in_channels=channel_sizes[0],
            out_channels=channel_sizes[1],
            depth=group_depths[1],
            stride=2,
        )
        self.group3 = self._make_group(
            in_channels=channel_sizes[1],
            out_channels=channel_sizes[2],
            depth=group_depths[2],
            stride=2,
        )

        self.avg_pool = hnn.HAvgPool2d(kernel_size=8, manifold=manifold)

        self.fc = hnn.HLinear(in_features=channel_sizes[2], out_features=10, manifold=manifold)

    def foward(self, x: ManifoldTensor) -> ManifoldTensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.avg_pool(x)
        # This will fail, need to add squeeze
        x = self.fc(x)
        return x

    def _make_group(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int = 1,
    ) -> nn.Sequential:
        if stride == 1:
            downsample = None
        else:
            downsample = hnn.HConvolution2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                manifold=self.manifold,
                stride=stride,
            )

        layers = [
            PoincareResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                manifold=self.manifold,
                stride=stride,
                downsample=downsample,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                PoincareResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    manifold=self.manifold,
                )
            )

        return nn.Sequential(*layers)
