<div align="center">

# ⚛️ VQC-EGNN

**Investigating the Regularization Properties of Variational Quantum Circuits (VQCs) within SE(3)-Equivariant Graph Neural Networks.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange?logo=pytorch)
![PennyLane](https://img.shields.io/badge/Quantum-PennyLane-magenta?logo=atom)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📜 Abstract
This project implements a **Hybrid Quantum-Classical Graph Neural Network** for protein structure modeling. By integrating an **SE(3)-Equivariant GNN** backbone with a **Variational Quantum Circuit (VQC)** projection head, we analyze how quantum-induced inductive biases—specifically entanglement and interference—alter the learning trajectory compared to classical MLPs.

---

## 📉 The Challenge: "Delta Learning"
Accurate exploration of the **Potential Energy Surface (PES)** is limited by the trade-off between speed and accuracy.

| **Method** | **Speed** | **Accuracy** | **Limitation** |
| :--- | :--- | :--- | :--- |
| **Ab Initio (DFT)** | 🐢 Very Slow | 🎯 High | Computationally prohibitive for large proteins. |
| **Classical MLPs** | 🐇 Fast | 📉 Low | Fails to capture **long-range electron correlations**. |

### 💡 My Approach
I address this by implementing a **Hybrid Quantum-Classical E(n)-Equivariant GNN** targeting the **Delta-Learning ($\Delta$-ML)** task.

> **Goal:** Predict the *residual error* between cheap semi-empirical approximations and high-fidelity DFT.

By processing geometric features through a **Variational Quantum Circuit (VQC)**, I aim to leverage the natural **entanglement capabilities** of quantum Hilbert spaces to capture electron correlations that classical message-passing layers fail to resolve.

---

## 🧠 Key Research Question
> *Does replacing the final classical dense layers with a parameterized quantum circuit introduce a distinct **inductive bias** that acts as an implicit regularizer?*

* **Hypothesis:** The limited Hilbert space connectivity and unitary nature of VQCs prevent the "fast memorization" often seen in over-parameterized classical networks.
* **Observation:** While classical heads exhibit rapid convergence (often overfitting), the **Quantum Head** demonstrates smoother loss landscapes and distinct generalization bounds.

---

## 🗺️ The Research Roadmap
This project is the culmination of a systematic study into Geometric Deep Learning and Quantum Computing.

```mermaid
graph TD
    subgraph P1 [Phase 1: Computer Vision]
        A[<b>CNNs / LeNet-5</b><br/>Grid Data]
    end

    subgraph P2 [Phase 2: Graph Theory]
        B[<b>Standard GNNs</b><br/>Irregular Structures]
    end

    subgraph P3 [Phase 3: Geometry & Physics]
        C[<b>Equivariant GNNs</b><br/>3D Symmetry]
    end

    subgraph P4 [Phase 4: Quantum Frontier]
        D[<b>Hybrid Quantum-Equivariant GNN</b><br/>Non-Classical Inductive Bias]
    end

    A -->|Limitation: Fails on 3D Rotation| B
    B -->|Limitation: Loses Spatial Coordinates| C
    C -->|Hypothesis: Classical Logic misses Electron Correlations| D

    style D fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    style P4 fill:#fff,stroke:#01579b,stroke-dasharray: 5 5
