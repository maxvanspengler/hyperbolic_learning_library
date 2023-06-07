# -*- coding: utf-8 -*-
"""
Training a Poincare ResNet
==========================

This is an implementation based on the Poincare Resnet paper, which can be found at:

- https://arxiv.org/abs/2303.14027

This tutorial assumes the availability of a GPU. Due to the complexity of hypebolic operations
we strongly advise to only run this tutorial with a GPU.

We will perform the following steps in order:

1. Define a hyperbolic manifold
2. Load and normalize the CIFAR10 training and test datasets using ``torchvision``
3. Define a Poincare ResNet
4. Define a loss function and optimizer
5. Train the network on the training data
6. Test the network on the test data

"""

#############################
# 1. Define the Poincare ball
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

from hypll.manifolds import Curvature, PoincareBall

# Making the curvature a learnable parameter is usually suboptimal but can
# make training smoother. An initial curvature of 0.1 has also been shown
# to help during training.
manifold = PoincareBall(c=Curvature(value=0.1, requires_grad=True))


###############################
# 2. Load and normalize CIFAR10
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

batch_size = 128

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)


###############################
# 3. Define a Poincare ResNet
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This implementation is based on the Poincare ResNet paper, which can
# be found at https://arxiv.org/abs/2303.14027 and which, in turn, is
# based on the original Euclidean implementation described in the paper
# Deep Residual Learning for Image Recognition by He et al. from 2015:
# https://arxiv.org/abs/1512.03385.


from typing import Optional

from torch import nn

from hypll import nn as hnn
from hypll.manifolds import PoincareBall
from hypll.tensors import ManifoldTensor


class PoincareResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        manifold: PoincareBall,
        stride: int = 1,
        downsample: Optional[nn.Sequential] = None,
    ):
        # We can replace each operation in the usual ResidualBlock by a manifold-agnostic
        # operation and supply the PoincareBall object to these operations.
        super().__init__()
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
        self.bn1 = hnn.HBatchNorm2d(features=out_channels, manifold=manifold)
        self.relu = hnn.HReLU(manifold=self.manifold)
        self.conv2 = hnn.HConvolution2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            manifold=manifold,
            padding=1,
        )
        self.bn2 = hnn.HBatchNorm2d(features=out_channels, manifold=manifold)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        # We replace the addition operation inside the skip connection by a Mobius addition.
        x = self.manifold.mobius_add(x, residual)
        x = self.relu(x)

        return x


class PoincareResNet(nn.Module):
    def __init__(
        self,
        channel_sizes: list[int],
        group_depths: list[int],
        manifold: PoincareBall,
    ):
        # For the Poincare ResNet itself we again replace each layer by a manifold-agnostic one
        # and supply the PoincareBall to each of these. We also replace the ResidualBlocks by
        # the manifold-agnostic one defined above.
        super().__init__()
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
        self.bn = hnn.HBatchNorm2d(features=channel_sizes[0], manifold=manifold)
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

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.avg_pool(x)
        x = self.fc(x.squeeze())
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


# Now, let's create a thin Poincare ResNet with channel sizes [4, 8, 16] and with a depth of 20
# layers.
net = PoincareResNet(
    channel_sizes=[4, 8, 16],
    group_depths=[3, 3, 3],
    manifold=manifold,
).cuda()


#########################################
# 4. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and RiemannianAdam optimizer.

criterion = nn.CrossEntropyLoss()
# net.parameters() includes the learnable curvature "c" of the manifold.
from hypll.optim import RiemannianAdam

optimizer = RiemannianAdam(net.parameters(), lr=0.001)


######################
# 5. Train the network
# ^^^^^^^^^^^^^^^^^^^^
# We simply have to loop over our data iterator, project the inputs onto the
# manifold, and feed them to the network and optimize.

from hypll.tensors import TangentTensor

for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].cuda(), data[1].cuda()

        # move the inputs to the manifold
        tangents = TangentTensor(data=inputs, man_dim=1, manifold=manifold)
        manifold_inputs = manifold.expmap(tangents)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(manifold_inputs)
        loss = criterion(outputs.tensor, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(f"[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}")
        running_loss = 0.0

print("Finished Training")


######################################
# 6. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].cuda(), data[1].cuda()

        # move the images to the manifold
        tangents = TangentTensor(data=images, man_dim=1, manifold=manifold)
        manifold_images = manifold.expmap(tangents)

        # calculate outputs by running images through the network
        outputs = net(manifold_images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.tensor, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")
