import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.egnn import EGNNLayer
from models.quantum_layer import QuantumLayer


class HybridEquivariantGNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.embedding = nn.Embedding(20, hidden_dim)

        self.egnn = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(3)
        ])

        # Compress graph embedding → small vector
        self.pre_quantum = nn.Linear(hidden_dim, 4)

        # Quantum layer
        self.quantum = QuantumLayer(n_qubits=4, n_layers=2)

        # Final classical head
        self.post_quantum = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x = self.embedding(data.x.long().squeeze(-1))
        pos = data.pos

        for layer in self.egnn:
            x, pos = layer(x, pos, data.edge_index, data.edge_attr)

        # Pool node features
        x = global_mean_pool(x, data.batch)

        # Classical → Quantum
        x = self.pre_quantum(x)
        x = self.quantum(x)

        return self.post_quantum(x)
