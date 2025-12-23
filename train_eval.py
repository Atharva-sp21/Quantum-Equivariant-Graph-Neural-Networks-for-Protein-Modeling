import torch
from torch_geometric.loader import DataLoader
from data.dataset import ProteinDataset

def train(model, epochs=100, name="model"):
    dataset = ProteinDataset(["data/1UBQ.pdb"])
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    losses = []

    for e in range(epochs):
        loss_sum = 0
        for batch in loader:
            pred = model(batch)
            loss = loss_fn(pred, batch.y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()

        losses.append(loss_sum)
        print(f"{name} | Epoch {e:03d} | Loss: {loss_sum:.4f}")

    return losses
