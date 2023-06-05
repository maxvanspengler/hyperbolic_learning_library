# Load the transitive closure of the mammals subset from the WordNet hierarchy.
# This is a directed graph that is transitively closed.
import json
import os
import networkx as nx


root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mammals_json_file = os.path.join(root, "data", "wordnet_mammals.json")
with open(mammals_json_file, "r") as json_file:
    graph_dict = json.load(json_file)

mammals_graph = nx.node_link_graph(graph_dict)


# Define the dataset that samples relations from hierarchy to be used during training.
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

    def __len__(self) -> int:
        return len(self.edges_list)
    
    def __getitem__(self, idx: int):
        rel = self.edges_list[idx]
        negative_target_nodes = list(
            self.mammals.nodes() - nx.descendants(self.mammals, rel[0]) - {rel[0]}
        )
        negative_target_sample_size = min(5, len(negative_target_nodes))
        negative_target_nodes_sample = random.sample(
            negative_target_nodes, negative_target_sample_size
        )

        edges = torch.tensor(
            [rel] + [[rel[0], neg] for neg in negative_target_nodes_sample]
        )

        negative_source_nodes = list(
            self.mammals.nodes() - nx.ancestors(self.mammals, rel[1]) - {rel[1]}
        )
        negative_source_sample_size = 10 - negative_target_sample_size
        negative_source_nodes_sample = random.sample(
            negative_source_nodes, negative_source_sample_size
        )

        edges = torch.cat(
            tensors=(
                edges,
                torch.tensor([[neg, rel[1]] for neg in negative_source_nodes_sample])
            ),
            dim=0,
        )

        edge_label_targets = torch.cat(
            tensors=[torch.ones(1).bool(), torch.zeros(10).bool()]
        )

        return edges, edge_label_targets
    
dataset = MammalsEmbeddingDataset(
    mammals=mammals_graph,
)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    

# Define the manifold on which the embeddings will be trained
from hll.manifolds import Curvature, PoincareBall


poincare_ball = PoincareBall(Curvature(1.0))


# Define the Poincare embedding model
import hll.nn as hnn


class PoincareEmbedding(hnn.HEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        manifold: PoincareBall,
    ):
        super().__init__(num_embeddings, embedding_dim, manifold)

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        embeddings = super().forward(edges)
        edge_distances = self.manifold.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
        return edge_distances
    

model = PoincareEmbedding(
    num_embeddings=len(mammals_graph.nodes()),
    embedding_dim=2,
    manifold=poincare_ball,
)

# Next, we define the poincare embedding loss as defined in the original paper <REF>
def poincare_embeddings_loss(
    dists: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    logits = dists.neg().exp()
    numerator = torch.where(condition=targets, input=logits, other=0).sum(dim=-1)
    denominator = logits.sum(dim=-1)
    loss = (numerator / denominator).log().mean().neg()
    return loss


# Now let's setup the "burn-in" phase of training
from hll.optim import RiemannianSGD


optimizer = RiemannianSGD(
    params=model.parameters(),
    lr=0.3 / 10,
)

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


# After burn-in we can start training our embeddings
optimizer = RiemannianSGD(
    params=model.parameters(),
    lr=0.3,
)

for epoch in range(300):
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
