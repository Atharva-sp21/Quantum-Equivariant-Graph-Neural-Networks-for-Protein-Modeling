import matplotlib.pyplot as plt
import pickle

# Load saved losses
with open("visualization/classical_losses.pkl", "rb") as f:
    classical = pickle.load(f)

with open("visualization/quantum_losses.pkl", "rb") as f:
    quantum = pickle.load(f)

plt.figure()
plt.plot(classical, label="EGNN + Classical Head")
plt.plot(quantum, label="EGNN + Quantum Head")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Log-Scale Training Dynamics")
plt.legend()
plt.show()
