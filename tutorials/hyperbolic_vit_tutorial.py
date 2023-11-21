# -*- coding: utf-8 -*-
"""
Training a Hyperbolic Vision Transformer
========================================

This implementation is based on the "Hyperbolic Vision Transformers: Combining Improvements in Metric Learning"
paper by Aleksandr Ermolov et al., which can be found at: 

- https://arxiv.org/pdf/2203.10833.pdf

Their implementation can be found at:

- https://github.com/htdt/hyp_metric

We will perform the following steps in order:

1. Initialize the manifold on which the embeddings will be trained
2. Load the CUB_200_2011 dataset using fastai
3. Initialize train and test dataloaders
4. Define the Hyperbolic Vision Transformer model
5. Define the pairwise cross-entropy loss function
6. Define the recall@k evaluation metric
7. Train the model on the training data
8. Test the model on the test data

Please make sure to install the following packages before running this script:
- fastai
- timm

"""

#######################################################
# 0. Set the device and random seed for reproducibility
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import random

import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

####################################################################
# 1. Initialize the manifold on which the embeddings will be trained
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from hypll.manifolds.euclidean import Euclidean
from hypll.manifolds.poincare_ball import Curvature, PoincareBall

# We can choose between the Poincaré ball model and Euclidean space.
do_hyperbolic = True

if do_hyperbolic:
    # We fix the curvature to 0.1 as in the paper (Section 3.2).
    manifold = PoincareBall(
        c=Curvature(value=0.1, constraining_strategy=lambda x: x, requires_grad=False)
    )
else:
    manifold = Euclidean()

###############################################
# 2. Load the CUB_200_2011 dataset using fastai
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import pandas as pd
import torchvision.transforms as transforms
from fastai.data.external import URLs, untar_data
from PIL import Image
from torch.utils.data import Dataset

# Download the dataset using fastai.
path = untar_data(URLs.CUB_200_2011) / "CUB_200_2011"

class CUB(Dataset):
    """Handles the downloaded CUB_200_2011 files."""

    def __init__(self, files_path, train, transform):
        self.files_path = files_path
        self.transform = transform

        self.targets = pd.read_csv(files_path / "image_class_labels.txt", header=None, sep=" ")
        self.targets.columns = ["id", "target"]
        self.targets = self.targets["target"].values

        self.train_test = pd.read_csv(files_path / "train_test_split.txt", header=None, sep=" ")
        self.train_test.columns = ["id", "is_train"]

        self.images = pd.read_csv(files_path / "images.txt", header=None, sep=" ")
        self.images.columns = ["id", "name"]

        mask = self.train_test.is_train.values == int(train)

        self.filenames = self.images.iloc[mask]
        self.targets = self.targets[mask]
        self.num_files = len(self.targets)

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        y = self.targets[index] - 1
        file_name = self.filenames.iloc[index, 1]
        path = self.files_path / "images" / file_name
        x = Image.open(path).convert("RGB")
        x = self.transform(x)
        return x, y

# We use the same resizing and data augmentation as in the paper (Section 3.2).
# Normalization values are taken from the original implementation.

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            224, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode("bicubic")
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
trainset = CUB(path, train=True, transform=train_transform)

test_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode("bicubic")),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
testset = CUB(path, train=False, transform=test_transform)

##########################################
# 3. Initialize train and test dataloaders
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import collections
import multiprocessing

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

# We need a dataloader that returns a requested number of images belonging to the same class.
# This will allow us to compute the pairwise cross-entropy loss.

class UniqueClassSampler(Sampler):
    """Custom sampler for creating batches with m_per_class samples per class."""

    def __init__(self, labels, m_per_class, seed):
        self.labels_to_indices = self.get_labels_to_indices(labels)
        self.labels = sorted(list(self.labels_to_indices.keys()))
        self.m_per_class = m_per_class
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return (
            np.max([len(v) for v in self.labels_to_indices.values()])
            * m_per_class
            * len(self.labels)
        )

    def set_epoch(self, epoch: int):
        # Update the epoch, so that the random permutation changes.
        self.epoch = epoch

    def get_labels_to_indices(self, labels: list):
        # Create a dictionary mapping each label to the indices in the dataset
        labels_to_indices = collections.defaultdict(list)
        for i, label in enumerate(labels):
            labels_to_indices[label].append(i)
        for k, v in labels_to_indices.items():
            labels_to_indices[k] = np.array(v, dtype=int)
        return labels_to_indices

    def __iter__(self):
        # Generate a random iteration order for the indices.
        # For example, if we have 3 classes (A,B,C) and m_per_class=2,
        # we could get the following indices: [A1, A2, C1, C2, B1, B2].
        idx_list = []
        g = torch.Generator()
        g.manual_seed(self.seed * 10000 + self.epoch)
        idx = torch.randperm(len(self.labels), generator=g).tolist()
        max_indices = np.max([len(v) for v in self.labels_to_indices.values()])
        for i in idx:
            t = self.labels_to_indices[self.labels[i]]
            idx_list.append(np.random.choice(t, size=self.m_per_class * max_indices))
        idx_list = np.stack(idx_list).reshape(len(self.labels), -1, self.m_per_class)
        idx_list = idx_list.transpose(1, 0, 2).reshape(-1).tolist()
        return iter(idx_list)

# First, define how many distinct classes to encounter per batch.
# The dataset has 200 classes in total, but the number we choose depends on the available memory.
n_sampled_classes = 128
# Then, define how many examples per class to encounter per batch.
# This number must be at least 2. 
m_per_class = 2
# Finally, the batch size is determined by our two choices above.
batch_size = n_sampled_classes * m_per_class

# Note that this sampler is only used during training.
sampler = UniqueClassSampler(labels=trainset.targets, m_per_class=m_per_class, seed=seed)

trainloader = DataLoader(
    dataset=trainset,
    sampler=sampler,
    batch_size=batch_size,
    num_workers=multiprocessing.cpu_count(),
    pin_memory=True,
    drop_last=True,
)

testloader = DataLoader(
    dataset=testset,
    batch_size=batch_size,
    num_workers=multiprocessing.cpu_count(),
    pin_memory=True,
    drop_last=False,
)

###################################################
# 4. Define the Hyperbolic Vision Transformer model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from typing import Union

import timm
from torch import nn

from hypll.tensors import ManifoldTensor, TangentTensor


class HyperbolicViT(nn.Module):
    def __init__(
        self,
        vit_name: str,
        vit_dim: int,
        emb_dim: int,
        manifold: Union[PoincareBall, Euclidean],
        clip_r: float,
        pretrained: bool = True,
    ):
        super().__init__()
        # We use the timm library to load a (pretrained) ViT backbone model.
        self.vit = timm.create_model(vit_name, pretrained=pretrained)
        # We remove the head of the model, since we won't need it.
        self.remove_head()
        self.fc = nn.Linear(vit_dim, emb_dim)
        # We freeze the linear projection for patch embeddings as in the paper (Section 3.2).
        self.freeze_patch_embed()
        self.manifold = manifold
        self.clip_r = clip_r

    def forward(self, x: torch.Tensor) -> ManifoldTensor:
        # We first encode the images using the ViT backbone.
        x = self.vit(x)
        if type(x) == tuple:
            x = x[0]
        # Then, we project the features into a lower-dimensional space.
        x = self.fc(x)
        # If we use a Euclidean manifold,
        # we simply normalize the features to lie on the unit sphere.
        if isinstance(self.manifold, Euclidean):
            x = F.normalize(x, p=2, dim=1)
            return ManifoldTensor(data=x, man_dim=1, manifold=self.manifold)
        # If we use a Poincaré ball model,
        # we (optionally) perform feature clipping (Section 2.4),
        if self.clip_r is not None:
            x = self.clip_features(x)
        # and then map the features to the manifold.
        tangents = TangentTensor(data=x, man_dim=1, manifold=self.manifold)
        return self.manifold.expmap(tangents)

    def clip_features(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
        fac = torch.minimum(torch.ones_like(x_norm), self.clip_r / x_norm)
        return x * fac

    def remove_head(self):
        names = set(x[0] for x in self.vit.named_children())
        target = {"head", "fc", "head_dist", "head_drop"}
        for x in names & target:
            self.vit.add_module(x, nn.Identity())

    def freeze_patch_embed(self):
        def fr(m):
            for param in m.parameters():
                param.requires_grad = False

        fr(self.vit.patch_embed)
        fr(self.vit.pos_drop)


# Initialize the model using the same hyperparameters as in the paper (Section 3.2).
hvit = HyperbolicViT(
    vit_name="vit_small_patch16_224", vit_dim=384, emb_dim=128, manifold=manifold, clip_r=2.3
).to(device)

####################################################
# 5. Define the pairwise cross-entropy loss function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch.nn.functional as F

# The function is described in Section 2.2 of the paper.

def pairwise_cross_entropy_loss(
    z_0: ManifoldTensor,
    z_1: ManifoldTensor,
    manifold: Union[PoincareBall, Euclidean],
    tau: float,
) -> torch.Tensor:
    dist_f = lambda x, y: -manifold.cdist(x.unsqueeze(0), y.unsqueeze(0))[0]
    num_classes = z_0.shape[0]
    target = torch.arange(num_classes, device=device)
    eye_mask = torch.eye(num_classes, device=device) * 1e9
    logits00 = dist_f(z_0, z_0) / tau - eye_mask
    logits01 = dist_f(z_0, z_1) / tau
    logits = torch.cat([logits01, logits00], dim=1)
    logits -= logits.max(1, keepdim=True)[0].detach()
    loss = F.cross_entropy(logits, target)
    return loss

# We use the same temperature as in the paper (Section 3.2).
criterion = lambda z_0, z_1: pairwise_cross_entropy_loss(z_0, z_1, manifold=manifold, tau=0.2)

##########################################
# 6. Define the recall@k evaluation metric
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Retrieval is performed by computing the distance between the query and all other embeddings.
# The top-k embeddings with the smallest distance are then used to compute the recall@k metric.

def eval_recall_k(
    k: int, model: nn.Module, dataloader: DataLoader, manifold: Union[PoincareBall, Euclidean]
):
    def get_embeddings():
        embs = []
        model.eval()
        for data in dataloader:
            x = data[0].to(device)
            with torch.no_grad():
                z = model(x)
            embs.append(z)
        embs = manifold.cat(embs, dim=0)
        model.train()
        return embs

    def get_recall_k(embs):
        dist_matrix = -manifold.cpu().cdist(embs.unsqueeze(0).cpu(), embs.unsqueeze(0).cpu())[0]
        dist_matrix = torch.nan_to_num(dist_matrix, nan=-torch.inf)
        targets = np.array(dataloader.dataset.targets)
        top_k = targets[dist_matrix.topk(1 + k).indices[:, 1:].numpy()]
        recall_k = np.mean([1 if t in top_k[i] else 0 for i, t in enumerate(targets)])
        return recall_k

    return get_recall_k(get_embeddings())


#########################################
# 7. Train the model on the training data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch.optim as optim

# We use the AdamW optimizer with the same hyperparameters as in the paper (Section 3.2).
optimizer = optim.AdamW(hvit.parameters(), lr=3e-5, weight_decay=0.01)

# Training for 5 epochs is enough to achieve good results.
num_epochs = 5
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    running_loss = 0.0
    num_samples = 0
    for d_i, data in enumerate(trainloader):
        # Get the inputs; data is a list of [inputs, labels]
        inputs = data[0].to(device)

        # Zero out the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        bs = len(inputs)
        z = hvit(inputs)
        z = ManifoldTensor(
            data=z.tensor.reshape((bs // m_per_class, m_per_class, -1)),
            man_dim=-1,
            manifold=z.manifold,
        )
        # The loss is computed pair-wise.
        loss = 0
        for i in range(m_per_class):
            for j in range(m_per_class):
                if i != j:
                    z_i = z[:, i]
                    z_j = z[:, j]
                    loss += criterion(z_i, z_j)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hvit.parameters(), 3)
        optimizer.step()

        # Print statistics every 10 batches.
        running_loss += loss.item() * bs
        num_samples += bs
        if d_i % 10 == 9:
            print(
                f"[Epoch {epoch + 1}, {d_i + 1:5d}/{len(trainloader)}] train loss: {running_loss/num_samples:.3f}"
            )
            running_loss = 0.0
            num_samples = 0

print("Finished training!")

####################################
# 8. Test the model on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Finally, let's see how well the model performs on the test data.

k = 5
recall_k = eval_recall_k(k, hvit, testloader, manifold)
print(f"Test recall@{k}: {recall_k:.3f}")

# Using the default hyperparameters in this tutorial,
# we are able to achieve a recall@5 of 0.945 for the Poincaré ball model,
# and a recall@5 of 0.916 for the Euclidean model.