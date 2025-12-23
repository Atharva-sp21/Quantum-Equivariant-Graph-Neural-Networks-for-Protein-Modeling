import torch
from torch_geometric.data import Data

def build_protein_graph(residue_types, coords, distance_threshold=8.0):
    """
    Builds a protein graph from residues and 3D coordinates.

    Args:
        residue_types: LongTensor [N]
        coords: FloatTensor [N, 3]
        distance_threshold: float (Å)

    Returns:
        PyTorch Geometric Data object
    """
    N = coords.size(0)
    edge_index = []
    edge_attr = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # Relative position vector
            r_ij = coords[j] - coords[i]
            dist = torch.norm(r_ij)

            if dist < distance_threshold:
                edge_index.append([i, j])

                # Edge feature = distance ONLY (scalar)
                edge_attr.append(dist.unsqueeze(0))  # [1]


    # Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # [2, E]
    edge_attr = torch.stack(edge_attr, dim=0)                    # [E, 4]

    data = Data(
        x=residue_types.unsqueeze(-1).float(),  # [N, 1]
        pos=coords,                             # [N, 3]
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return data

# Edge feature now = [dx, dy, dz, distance]r_ij = coords[j] - coords[i]
