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

#################################
# Define hyperbolic manifold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from hypll.manifolds.poincare_ball import Curvature, PoincareBall

manifold = PoincareBall(c=Curvature(value=0.1, requires_grad=False))

#################################
# Samplers for positive pairs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import collections
import multiprocessing

import torchvision.transforms as transforms
from fastai.data.external import URLs, untar_data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision.datasets import ImageFolder


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


# Obtained from original implementation (https://github.com/htdt/hyp_metric/blob/master/sampler.py)
# As far as I understand, it only allows for one batch per epoch, containing m_per_class examples per class


class UniqueClassSampler1(Sampler):
    def __init__(self, labels, m_per_class, seed):
        self.labels_to_indices = get_labels_to_indices(labels)
        self.labels = sorted(list(self.labels_to_indices.keys()))
        self.m_per_class = m_per_class
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.labels) * self.m_per_class

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        idx_list = []
        g = torch.Generator()
        g.manual_seed(self.seed * 10000 + self.epoch)
        idx = torch.randperm(len(self.labels), generator=g).tolist()
        for i in idx:
            t = self.labels_to_indices[self.labels[i]]
            replace = len(t) < self.m_per_class
            idx_list += np.random.choice(t, size=self.m_per_class, replace=replace).tolist()
        return iter(idx_list)


# An alternative sampler, I made some changes to go through the whole dataset in one epoch


class UniqueClassSampler2(Sampler):
    def __init__(self, labels, m_per_class, seed):
        self.length = len(labels) * m_per_class
        self.labels_to_indices = get_labels_to_indices(labels)
        self.labels = sorted(list(self.labels_to_indices.keys()))
        self.m_per_class = m_per_class
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
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


######################################
# Imagenette datasets and dataloaders
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

path = untar_data(URLs.IMAGENETTE_320)
imagenette_classes = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode("bicubic")),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
trainset = ImageFolder(path / "train", transform=train_transform)
trainset.classes = imagenette_classes

val_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
valset = ImageFolder(path / "val", transform=val_transform)
valset.classes = imagenette_classes

n_sampled_labels = 10
m_per_class = 2
batch_size = n_sampled_labels * m_per_class

sampler = UniqueClassSampler2(labels=trainset.targets, m_per_class=m_per_class, seed=seed)

trainloader = DataLoader(
    dataset=trainset,
    sampler=sampler,
    batch_size=batch_size,
    num_workers=multiprocessing.cpu_count(),
    pin_memory=True,
    drop_last=True,
)

valloader = DataLoader(
    dataset=valset,
    batch_size=batch_size,
    num_workers=multiprocessing.cpu_count(),
    pin_memory=True,
    drop_last=False,
)

#######################################
# Define Hyperbolic Vision Transformer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


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
        self.freeze_patch_embed()
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

    def freeze_patch_embed(self):
        def fr(m):
            for param in m.parameters():
                param.requires_grad = False

        fr(self.vit.patch_embed)
        fr(self.vit.pos_drop)


hvit = HyperbolicVIT(
    vit_name="vit_small_patch16_224", vit_dim=384, emb_dim=128, manifold=manifold, clip_r=2.3
).to(device)

#######################################
# Define loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

################################
# Evaluation through recall@k
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

k = 100


def eval_recall_k(k):
    def get_embeddings():
        embs = []
        hvit.eval()
        for data in valloader:
            x = data[0].to(device)
            with torch.no_grad():
                z = hvit(x).tensor
            embs.append(z)
        embs = torch.cat(embs, dim=0)
        hvit.train()
        return embs

    def get_recall_k(embs):
        embs_ = ManifoldTensor(embs, manifold=manifold)
        dist_matrix = -manifold.dist_matrix(embs_, embs_)
        dist_matrix.diagonal().fill_(torch.inf)
        dist_matrix = torch.nan_to_num(dist_matrix, nan=torch.inf)
        dist_matrix = -dist_matrix

        targets = np.array(valset.targets)
        top_k = targets[dist_matrix.topk(k).indices.cpu().numpy()]
        retrieved = (top_k == targets[:, None]).sum(-1)
        relevant = np.bincount(targets)[targets]
        recall_k = np.mean(retrieved / relevant)
        return recall_k

    return get_recall_k(get_embeddings())


######################
# Train the network
# ^^^^^^^^^^^^^^^^^^^^

num_epochs = 100
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    running_loss = 0.0
    num_samples = 0
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
        torch.nn.utils.clip_grad_norm_(hvit.parameters(), 3)
        optimizer.step()

        # print statistics
        running_loss += loss.item() * bs
        num_samples += bs
        if d_i % 10 == 9:
            recall_k = eval_recall_k(k)
            print(
                f"[Epoch {epoch + 1}, {d_i + 1:5d}] loss: {running_loss/num_samples:.3f} recall@{k}: {recall_k:.3f}"
            )
            running_loss = 0.0

print("Finished Training")
