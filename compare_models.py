import matplotlib.pyplot as plt
import pickle
from models.egnn_classical import EGNNClassical
from models.hybrid_equivariant_gnn import HybridEquivariantGNN
from train_eval import train

# Train models
classical_losses = train(EGNNClassical(), name="Classical")
quantum_losses = train(HybridEquivariantGNN(), name="Quantum")

with open("classical_losses.pkl", "wb") as f:
    pickle.dump(classical_losses, f)

with open("quantum_losses.pkl", "wb") as f:
    pickle.dump(quantum_losses, f)

# Plot
plt.figure()
plt.plot(classical_losses, label="EGNN + Classical Head")
plt.plot(quantum_losses, label="EGNN + Quantum Head")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Classical vs Quantum Head Training Dynamics")
plt.show()
