import matplotlib.pyplot as plt
import pickle

def ema(x, alpha=0.2):
    y = [x[0]]
    for i in range(1, len(x)):
        y.append(alpha * x[i] + (1 - alpha) * y[-1])
    return y

with open("visualization/classical_losses.pkl", "rb") as f:
    classical = pickle.load(f)

with open("visualization/quantum_losses.pkl", "rb") as f:
    quantum = pickle.load(f)

plt.figure()
plt.plot(ema(classical), label="Classical (EMA)")
plt.plot(ema(quantum), label="Quantum (EMA)")
plt.xlabel("Epoch")
plt.ylabel("Smoothed Loss")
plt.title("Smoothed Training Dynamics (EMA)")
plt.legend()
plt.show()
