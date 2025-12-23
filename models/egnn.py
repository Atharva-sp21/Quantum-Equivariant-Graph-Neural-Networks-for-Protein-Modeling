import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EGNNLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr="add")

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, x, pos, edge_index, edge_attr):
        # propagate returns aggregated node messages
        m_agg, coord_agg = self.propagate(
            edge_index, x=x, pos=pos, edge_attr=edge_attr
        )

        # node update
        x = x + self.node_mlp(m_agg)
        pos = pos + coord_agg

        return x, pos

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        # scalar edge message
        m_ij = self.edge_mlp(
            torch.cat([x_i, x_j, edge_attr], dim=-1)
        )

        # equivariant coordinate update
        delta = pos_i - pos_j
        coord_update = self.coord_mlp(m_ij) * delta

        return m_ij, coord_update

    def aggregate(self, inputs, index):
        m_ij, coord_ij = inputs

        # aggregate edge → node
        m_out = torch.zeros(
            (index.max() + 1, m_ij.size(-1)),
            device=m_ij.device
        )
        coord_out = torch.zeros(
            (index.max() + 1, coord_ij.size(-1)),
            device=coord_ij.device
        )

        m_out.index_add_(0, index, m_ij)
        coord_out.index_add_(0, index, coord_ij)

        return m_out, coord_out
