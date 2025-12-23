import torch

# Standard 20 amino acids
AMINO_ACIDS = {
    "ALA": 0, "CYS": 1, "ASP": 2, "GLU": 3, "PHE": 4,
    "GLY": 5, "HIS": 6, "ILE": 7, "LYS": 8, "LEU": 9,
    "MET": 10, "ASN": 11, "PRO": 12, "GLN": 13, "ARG": 14,
    "SER": 15, "THR": 16, "VAL": 17, "TRP": 18, "TYR": 19
}

def parse_pdb(pdb_path):
    """
    Parses a PDB file and extracts:
    - amino acid indices
    - Cα coordinates

    Returns:
        residue_types: torch.LongTensor [N]
        coords: torch.FloatTensor  [N, 3]
    """
    residue_types = []
    coords = []

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            residue_name = line[17:20].strip()
            if residue_name not in AMINO_ACIDS:
                continue

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            residue_types.append(AMINO_ACIDS[residue_name])
            coords.append([x, y, z])

    residue_types = torch.tensor(residue_types, dtype=torch.long)
    coords = torch.tensor(coords, dtype=torch.float)

    return residue_types, coords


if __name__ == "__main__":
    pdb_file = "1UBQ.pdb"  # replace with real path
    residue_types, coords = parse_pdb(pdb_file)

    print("Number of residues:", len(residue_types))
    print("First 5 residue types:", residue_types[:5])
    print("First 5 coordinates:\n", coords[:5])
