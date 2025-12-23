import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from models.egnn import EGNNLayer
from models.classical_head import ClassicalHead

class EGNNClassical(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.embedding = nn.Embedding(20, hidden_dim)
        self.egnn = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(3)])
        self.head = ClassicalHead(hidden_dim)

    def forward(self, data):
        x = self.embedding(data.x.long().squeeze(-1))
        pos = data.pos

        for layer in self.egnn:
            x, pos = layer(x, pos, data.edge_index, data.edge_attr)

        x = global_mean_pool(x, data.batch)
        return self.head(x)
