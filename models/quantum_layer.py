import pennylane as qml
import torch
import torch.nn as nn

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # 1. Encoding (Angle Embedding)
            # Reshape inputs to standard form if needed
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

            # 2. Variational Layers (Strongly Entangling is standard/better than manual CNOTs)
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # 3. Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # The weight shape for StronglyEntanglingLayers is (n_layers, n_qubits, 3)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # qnn.TorchLayer 
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # x shape: [batch_size, n_qubits]
        return self.q_layer(x)