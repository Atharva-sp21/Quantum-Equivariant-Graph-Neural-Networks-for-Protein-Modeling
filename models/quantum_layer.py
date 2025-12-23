import torch
import pennylane as qml
from torch import nn

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # Encode classical data -> superposition ( quantum state )
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[l, i], wires=i) #trainable parameters
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1]) #entagling layer

            # Back to classical data
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        self.weights = nn.Parameter(
            0.01 * torch.randn(n_layers, n_qubits)
        )

    def forward(self, x):
        """
        x: [B, n_qubits]
        """
        outputs = []
        for i in range(x.shape[0]):
            out = self.circuit(x[i], self.weights)
            outputs.append(torch.stack(out))

        return torch.stack(outputs).float()
