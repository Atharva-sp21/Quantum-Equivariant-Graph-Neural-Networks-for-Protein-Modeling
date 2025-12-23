import matplotlib.pyplot as plt
import pickle
import numpy as np

window = 5

with open("visualization/classical_losses.pkl", "rb") as f:
    classical = np.array(pickle.load(f))

with open("visualization/quantum_losses.pkl", "rb") as f:
    quantum = np.array(pickle.load(f))

def rolling_var(x, w):
    return [np.var(x[i:i+w]) for i in range(len(x)-w)]

plt.figure()
plt.plot(rolling_var(classical, window), label="Classical Variance")
plt.plot(rolling_var(quantum, window), label="Quantum Variance")
plt.xlabel("Epoch")
plt.ylabel("Rolling Variance")
plt.title("Optimization Stability (Variance)")
plt.legend()
plt.show()
