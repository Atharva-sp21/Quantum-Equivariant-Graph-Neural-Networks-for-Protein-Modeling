import torch
from torch.utils.data import Dataset
from data.parse_pdb import parse_pdb
from data.protein_graph import build_protein_graph

class ProteinDataset(Dataset):
    def __init__(self, pdb_files):
        """
        Args:
            pdb_files: list of paths to .pdb files
        """
        self.pdb_files = pdb_files

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        pdb_path = self.pdb_files[idx]

        residue_types, coords = parse_pdb(pdb_path)
        graph = build_protein_graph(residue_types, coords)

        # Toy regression label: protein length
        label = torch.tensor([[graph.num_nodes]], dtype=torch.float)

        graph.y = label
        return graph
