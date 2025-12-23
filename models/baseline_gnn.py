import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class BaselineGNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        # Node embedding: amino acid ID -> vector
        self.embedding = nn.Embedding(num_embeddings=20, embedding_dim=hidden_dim)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # regression output
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # x is [N,1] → convert to integer indices
        x = x.long().squeeze(-1)
        x = self.embedding(x)

        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Pool node features → graph feature
        x = global_mean_pool(x, batch)

        out = self.mlp(x)
        return out
