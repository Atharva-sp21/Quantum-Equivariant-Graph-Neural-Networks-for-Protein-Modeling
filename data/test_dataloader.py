from torch_geometric.loader import DataLoader
from dataset import ProteinDataset

dataset = ProteinDataset(["1UBQ.pdb"])
loader = DataLoader(dataset, batch_size=1, shuffle=True)

for batch in loader:
    print(batch)
    print("Batch y:", batch.y)
