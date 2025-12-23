import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool

class GeometryGNN(MessagePassing):
    def __init__(self, hidden_dim=64):
        super().__init__(aggr="add")

        self.embedding = nn.Embedding(20, hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.embedding(x.long().squeeze(-1))
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)
        return self.readout(x)

    def message(self, x_j, edge_attr):
        edge_feat = self.edge_mlp(edge_attr)
        return x_j + edge_feat
