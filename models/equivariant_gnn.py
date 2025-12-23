import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from models.egnn import EGNNLayer


class EquivariantGNN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()

        self.embedding = nn.Embedding(20, hidden_dim)

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(num_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x = self.embedding(data.x.long().squeeze(-1))
        pos = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr)

        x = global_mean_pool(x, data.batch)
        return self.readout(x)
