import torch
from torch_geometric.loader import DataLoader

from data.dataset import ProteinDataset
from models.hybrid_equivariant_gnn import HybridEquivariantGNN

dataset = ProteinDataset(["data/1UBQ.pdb"])
loader = DataLoader(dataset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridEquivariantGNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(100):
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)

        pred = model(batch)
        loss = loss_fn(pred, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f}")
