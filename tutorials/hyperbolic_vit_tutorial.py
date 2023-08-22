import random

import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 0

torch.use_deterministic_algorithms(True)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# --------------------

from hypll.manifolds.poincare_ball import Curvature, PoincareBall

manifold = PoincareBall(c=Curvature(value=0.1, requires_grad=False))

# --------------------

import collections

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler


def get_labels_to_indices(labels: list) -> dict[str, np.ndarray]:
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices that will be used to index into self.dataset
    """
    labels_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for k, v in labels_to_indices.items():
        labels_to_indices[k] = np.array(v, dtype=int)
    return labels_to_indices


class UniqueClassSampler(Sampler):
    def __init__(self, labels: list, m_per_class: int, seed: int):
        self.length = len(labels) * m_per_class
        self.labels_to_indices = get_labels_to_indices(labels)
        self.labels = sorted(list(self.labels_to_indices.keys()))
        self.m_per_class = m_per_class
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.length

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        idx_list = []
        g = torch.Generator()
        g.manual_seed(self.seed * 10000 + self.epoch)
        idx = torch.randperm(len(self.labels), generator=g).tolist()
        max_indices = np.max(list(self.labels_to_indices.values()))
        for i in idx:
            t = self.labels_to_indices[self.labels[i]]
            idx_list.append(np.random.choice(t, size=self.m_per_class * max_indices))
        idx_list = np.stack(idx_list).reshape(len(self.labels), -1, self.m_per_class)
        idx_list = idx_list.transpose(1, 0, 2).reshape(-1).tolist()
        return iter(idx_list)


# --------------------

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

n_sampled_labels = 10
m_per_class = 2
batch_size = n_sampled_labels * m_per_class

sampler = UniqueClassSampler(labels=trainset.targets, m_per_class=m_per_class, seed=seed)

trainloader = DataLoader(dataset=trainset, sampler=sampler, batch_size=batch_size)

# --------------------

import timm
from torch import nn

from hypll.tensors import ManifoldTensor, TangentTensor


class HyperbolicVIT(nn.Module):
    def __init__(
        self,
        vit_name: str,
        vit_dim: int,
        emb_dim: int,
        manifold: PoincareBall,
        clip_r: float,
        pretrained: bool = True,
    ):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=pretrained)
        self.remove_head()
        self.fc = nn.Linear(vit_dim, emb_dim)
        self.manifold = manifold
        self.clip_r = clip_r

    def forward(self, x: torch.Tensor) -> ManifoldTensor:
        x = self.vit(x)
        if type(x) == tuple:
            x = x[0]
        x = self.fc(x)
        if self.clip_r is not None:
            x = self.clip_features(x)
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


hvit = HyperbolicVIT(
    vit_name="vit_small_patch16_224", vit_dim=384, emb_dim=128, manifold=manifold, clip_r=2.3
).to(device)

# --------------------

import torch.nn.functional as F
import torch.optim as optim


def pairwise_cross_entropy_loss(
    z_0: ManifoldTensor,
    z_1: ManifoldTensor,
    manifold: PoincareBall,
    tau: float,
) -> torch.Tensor:
    dist_f = lambda x, y: manifold.dist_matrix(x, y)
    batch_size = z_0.shape[0]
    target = torch.arange(batch_size, device=device)
    eye_mask = torch.eye(batch_size, device=device) * 1e9
    logits00 = dist_f(z_0, z_0) / tau - eye_mask
    logits01 = dist_f(z_0, z_1) / tau
    logits = torch.cat([logits01, logits00], dim=1)
    logits -= logits.max(1, keepdim=True)[0].detach()
    loss = F.cross_entropy(logits, target)
    return loss


criterion = lambda z_0, z_1: pairwise_cross_entropy_loss(z_0, z_1, manifold=manifold, tau=0.2)
optimizer = optim.AdamW(hvit.parameters(), lr=3e-5, weight_decay=0.01)

# --------------------

num_epochs = 2
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    running_loss = 0.0
    for d_i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data[0].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        bs = len(inputs)
        z = hvit(inputs).tensor.view(bs // m_per_class, m_per_class, -1)
        loss = 0
        for i in range(m_per_class):
            for j in range(m_per_class):
                if i != j:
                    z_i = ManifoldTensor(z[:, i], manifold=manifold)
                    z_j = ManifoldTensor(z[:, j], manifold=manifold)
                    loss += criterion(z_i, z_j)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if d_i % 10 == 9:  # print every 10 mini-batches
            print(f"[{epoch + 1}, {d_i + 1:5d}] loss: {loss.item()/10:.3f}")
            running_loss = 0.0

print("Finished Training")
