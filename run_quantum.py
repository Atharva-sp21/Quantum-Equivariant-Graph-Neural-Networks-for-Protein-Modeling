from models.hybrid_equivariant_gnn import HybridEquivariantGNN
from train_eval import train

model = HybridEquivariantGNN()
train(model)
