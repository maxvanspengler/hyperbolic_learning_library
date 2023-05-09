# -*- coding: utf-8 -*-
"""
Training a Hyperbolic Classifier
================================

This is an adaptation of torchvision's tutorial "Training a Classifier" to 
hyperbolic space. The original tutorial can be found here:

- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Training a Hyperbolic Image Classifier
--------------------------------------

We will do the following steps in order:

1. Load and normalize the CIFAR10 training and test datasets using ``torchvision``
2. Define a hyperbolic manifold
3. Define a hyperbolic Convolutional Neural Network
4. Define a loss function and optimizer
5. Train the network on the training data
6. Test the network on the test data

1. Load and normalize CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 4

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

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


########################################################################
# 2. Define a hyperbolic manifold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We use the Poincaré ball model for the purposes of this tutorial.


from hypdl.manifolds import PoincareBall
from hypdl.tensors import ManifoldTensor

# Setting the curvature of the ball to 0.1 and making it a learnable parameter
# is usually suboptimal but can make training smoother.
manifold = PoincareBall(c=0.1, learnable=True)


def get_manifold_tensor(tensor: torch.Tensor, dim: int) -> ManifoldTensor:
    # TODO: Can expmap0 return a ManifoldTensor directly?
    return ManifoldTensor(data=manifold.expmap0(v=tensor, dim=dim), manifold=manifold, man_dim=dim)


########################################################################
# 3. Define a hyperbolic Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's rebuild the convolutional neural network from torchvision's tutorial
# using hyperbolic modules.


from torch import nn

from hypdl import nn as hnn


# TODO: Add base class hnn.Module?
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = hnn.HConvolution2d(
            in_channels=3, out_channels=6, kernel_size=5, manifold=manifold
        )
        self.pool = hnn.HMaxPool2d(kernel_size=2, manifold=manifold, stride=2)
        self.conv2 = hnn.HConvolution2d(
            in_channels=6, out_channels=16, kernel_size=5, manifold=manifold
        )
        self.fc1 = hnn.HLinear(in_features=16 * 5 * 5, out_features=120, manifold=manifold)
        self.fc2 = hnn.HLinear(in_features=120, out_features=84, manifold=manifold)
        self.fc3 = hnn.HLinear(in_features=84, out_features=10, manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

########################################################################
# 4. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and Adam optimizer. Adam is
# preferred because hyperbolic linear layers can sometimes have training
# difficulties early on due to poor initialization.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# net.parameters() includes the learnable curvature "c" of the manifold.
optimizer = optim.Adam(net.parameters())


########################################################################
# 5. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, project the inputs onto the
# manifold, and feed them to the network and optimize.

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        manifold_inputs = get_manifold_tensor(tensor=inputs, dim=1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(manifold_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")


########################################################################
# Let's quickly save our trained model:

PATH = "./cifar_net.pth"
torch.save(net.state_dict(), PATH)


########################################################################
# Next, let's load back in our saved model (note: saving and re-loading the model
# wasn't necessary here, we only did it to illustrate how to do so):

net = Net()
net.load_state_dict(torch.load(PATH))

########################################################################
# 6. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        manifold_images = get_manifold_tensor(tensor=images, dim=1)
        # calculate outputs by running images through the network
        outputs = net(manifold_images)
        # the class with the highest energy is what we choose as prediction
        # TODO: Make torch.max and similar functions work on ManifoldTensor
        _, predicted = torch.max(outputs.tensor, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")


########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        manifold_images = get_manifold_tensor(tensor=images, dim=1)
        outputs = net(manifold_images)
        # TODO: Make torch.max and similar functions work on ManifoldTensor
        _, predictions = torch.max(outputs.tensor, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")


########################################################################
#
# Training on GPU
# ----------------
# Just like how you transfer a Tensor onto the GPU, you transfer the neural
# net onto the GPU.
#
# Let's first define our device as the first visible cuda device if we have
# CUDA available:
#
# .. code:: python
#
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#
# Assuming that we are on a CUDA machine, this should print a CUDA device:
#
# .. code:: python
#
#     print(device)
#
#
# The rest of this section assumes that ``device`` is a CUDA device.
#
# Then these methods will recursively go over all modules and convert their
# parameters and buffers to CUDA tensors:
#
# .. code:: python
#
#     net.to(device)
#
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# .. code:: python
#
#         inputs, labels = data[0].to(device), data[1].to(device)
#
#
# **Goals achieved**:
#
# - Train a small hyperbolic neural network to classify images.
#
