# -*- coding: utf-8 -*-
"""
Training Poincare embeddings for the mammals subset from WordNet
================================================================

This implementation is based on the "Poincare Embeddings for Learning Hierarchical Representations"
paper by Maximilian Nickel and Douwe Kiela, which can be found at: 

- https://arxiv.org/pdf/1705.08039.pdf

Their implementation can be found at:

- https://github.com/facebookresearch/poincare-embeddings

We will perform the following steps in order:

1. Load the mammals subset from the WordNet hierarchy using NetworkX
2. Create a dataset containing the graph from which we can sample
3. Initialize the Poincare ball on which the embeddings will be trained
4. Define the Poincare embedding model
5. Define the Poincare embedding loss function
6. Perform a few "burn-in" training epochs with reduced learning rate

"""

######################################################################
# 1. Load the mammals subset from the WordNet hierarchy using NetworkX
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import json
import os

import networkx as nx

# If you have stored the wordnet_mammals.json file differently that the definition here,
# adjust the mammals_json_file string to correctly point to this json file. The file itself can
# be found in the repository under tutorials/data/wordnet_mammals.json.
root = os.path.dirname(os.path.abspath(__file__))
mammals_json_file = os.path.join(root, "data", "wordnet_mammals.json")
with open(mammals_json_file, "r") as json_file:
    graph_dict = json.load(json_file)

mammals_graph = nx.node_link_graph(graph_dict)


###################################################################
# 2. Create a dataset containing the graph from which we can sample
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import random

import torch
from torch.utils.data import DataLoader, Dataset


class MammalsEmbeddingDataset(Dataset):
    def __init__(
        self,
        mammals: nx.DiGraph,
    ):
        super().__init__()
        self.mammals = mammals
        self.edges_list = list(mammals.edges())

    # This Dataset object has a sample for each edge in the graph.
    def __len__(self) -> int:
        return len(self.edges_list)

    def __getitem__(self, idx: int):
        # For each existing edge in the graph we choose 10 fake or negative edges, which we build
        # from the idx-th existing edge. So, first we grab this edge from the graph.
        rel = self.edges_list[idx]

        # Next, we take our source node rel[0] and see which nodes in the graph are not a child of
        # this node.
        negative_target_nodes = list(
            self.mammals.nodes() - nx.descendants(self.mammals, rel[0]) - {rel[0]}
        )

        # Then, we sample at most 5 of these negative target nodes...
        negative_target_sample_size = min(5, len(negative_target_nodes))
        negative_target_nodes_sample = random.sample(
            negative_target_nodes, negative_target_sample_size
        )

        # and add these to a tensor which will be used as input for our embedding model.
        edges = torch.tensor([rel] + [[rel[0], neg] for neg in negative_target_nodes_sample])

        # Next, we do the same with our target node rel[1], but now where we sample from nodes
        # which aren't a parent of it.
        negative_source_nodes = list(
            self.mammals.nodes() - nx.ancestors(self.mammals, rel[1]) - {rel[1]}
        )

        # We sample from these negative source nodes until we have a total of 10 negative edges...
        negative_source_sample_size = 10 - negative_target_sample_size
        negative_source_nodes_sample = random.sample(
            negative_source_nodes, negative_source_sample_size
        )

        # and add these to the tensor that we created above.
        edges = torch.cat(
            tensors=(edges, torch.tensor([[neg, rel[1]] for neg in negative_source_nodes_sample])),
            dim=0,
        )

        # Lastly, we create a tensor containing the labels of the edges, indicating whether it's a
        # True or a False edge.
        edge_label_targets = torch.cat(tensors=[torch.ones(1).bool(), torch.zeros(10).bool()])

        return edges, edge_label_targets


# Now, we construct the dataset.
dataset = MammalsEmbeddingDataset(
    mammals=mammals_graph,
)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


#########################################################################
# 3. Initialize the Poincare ball on which the embeddings will be trained
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from hypll.manifolds.poincare_ball import Curvature, PoincareBall

poincare_ball = PoincareBall(Curvature(1.0))


########################################
# 4. Define the Poincare embedding model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import hypll.nn as hnn


class PoincareEmbedding(hnn.HEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        manifold: PoincareBall,
    ):
        super().__init__(num_embeddings, embedding_dim, manifold)

    # The model outputs the distances between the nodes involved in the input edges as these are
    # used to compute the loss.
    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        embeddings = super().forward(edges)
        edge_distances = self.manifold.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
        return edge_distances


# We want to embed every node into a 2-dimensional Poincare ball.
model = PoincareEmbedding(
    num_embeddings=len(mammals_graph.nodes()),
    embedding_dim=2,
    manifold=poincare_ball,
)


################################################
# 5. Define the Poincare embedding loss function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# This function is given in equation (5) of the Poincare Embeddings paper.
def poincare_embeddings_loss(dists: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = dists.neg().exp()
    numerator = torch.where(condition=targets, input=logits, other=0).sum(dim=-1)
    denominator = logits.sum(dim=-1)
    loss = (numerator / denominator).log().mean().neg()
    return loss


#######################################################################
# 6. Perform a few "burn-in" training epochs with reduced learning rate
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from hypll.optim import RiemannianSGD

# The learning rate of 0.3 is dived by 10 during burn-in.
optimizer = RiemannianSGD(
    params=model.parameters(),
    lr=0.3 / 10,
)

# Perform training as we would usually.
for epoch in range(10):
    average_loss = 0
    for idx, (edges, edge_label_targets) in enumerate(dataloader):
        optimizer.zero_grad()

        dists = model(edges)
        loss = poincare_embeddings_loss(dists=dists, targets=edge_label_targets)
        loss.backward()
        optimizer.step()

        average_loss += loss

    average_loss /= len(dataloader)
    print(f"Burn-in epoch {epoch} loss: {average_loss}")


#########################
# 6. Train the embeddings
# ^^^^^^^^^^^^^^^^^^^^^^^

# Now we use the actual learning rate 0.3.
optimizer = RiemannianSGD(
    params=model.parameters(),
    lr=0.3,
)

for epoch in range(30):
    average_loss = 0
    for idx, (edges, edge_label_targets) in enumerate(dataloader):
        optimizer.zero_grad()

        dists = model(edges)
        loss = poincare_embeddings_loss(dists=dists, targets=edge_label_targets)
        loss.backward()
        optimizer.step()

        average_loss += loss

    average_loss /= len(dataloader)
    print(f"Epoch {epoch} loss: {average_loss}")

# You have now trained your own Poincare Embeddings!
