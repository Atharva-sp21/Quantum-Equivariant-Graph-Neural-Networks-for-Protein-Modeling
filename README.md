# Quantum-Inductive-Bias-EGNN

**An investigation into the regularization properties of Variational Quantum Circuits (VQCs) within SE(3)-Equivariant Graph Neural Networks.**

## 📜 Abstract
This project implements a **Hybrid Quantum-Classical Graph Neural Network** for protein structure modeling. By integrating an SE(3)-Equivariant GNN backbone with a Variational Quantum Circuit (VQC) projection head, we analyze how quantum-induced inductive biases—specifically entanglement and interference—alter the learning trajectory compared to classical MLPs.
## The Problem
Accurate exploration of the Potential Energy Surface (PES) for protein-ligand interactions is currently limited by the computational cost of Ab Initio methods (like DFT). While Machine Learning Potentials (MLPs) offer a speedup, classical architectures often struggle to model complex many-body electron correlation effects efficiently, leading to poor generalization outside the training distribution.
## My Approach
I address this by implementing a Hybrid Quantum-Classical E(n)-Equivariant GNN. By targeting the Delta-Learning ($\Delta$-ML) task—predicting the residual error between semi-empirical approximations and high-fidelity DFT—and processing geometric features through a Variational Quantum Circuit (VQC), I aim to leverage the natural entanglement capabilities of quantum Hilbert spaces to capture electron correlations that classical message-passing layers fail to resolve.
## 🧠 Key Research Question
Does replacing the final classical dense layers with a parameterized quantum circuit introduce a distinct **inductive bias** that acts as an implicit regularizer?

* **Hypothesis:** The limited Hilbert space connectivity and unitary nature of VQCs prevent the "fast memorization" often seen in over-parameterized classical networks.
* **Observation:** While classical heads exhibit rapid convergence (often overfitting), the Quantumsem Head demonstrates smoother loss landscapes and distinct generalization bounds.
## 🔬 Verification: Equivariance Test
A critical requirement for physically grounded molecular models is **SE(3)-Equivariance**. The model's energy prediction must remain identical regardless of how the protein is rotated in 3D space.

We validated this property by comparing the model's output for a random protein graph against a randomly rotated version ($R \in SO(3)$).

<p align="center">
  <img src="assets/equivariance_test.png" alt="Equivariance Verification Output" width="650">
  <br>
  <em>Figure: Validation script output confirming zero deviation (Difference: 0.00e+00) between original and rotated inputs, proving that the Quantum VQC layer preserves the geometric symmetries of the EGNN backbone.</em>
</p>

## 📉 The Research Roadmap
This project is the culmination of a systematic study into Geometric Deep Learning and Quantum Computing.

```mermaid
graph TD
    subgraph P1 [Phase 1: Computer Vision]
        A[<b>CNNs / LeNet-5</b><br/><i>Grid Data</i>]
    end

    subgraph P2 [Phase 2: Graph Theory]
        B[<b>Standard GNNs</b><br/><i>Irregular Structures</i>]
    end

    subgraph P3 [Phase 3: Geometry & Physics]
        C[<b>E n -Equivariant GNNs</b><br/><i>3D Symmetry</i>]
    end

    subgraph P4 [Phase 4: Quantum Frontier]
        D[<b>Hybrid Quantum-Equivariant GNN</b><br/><i>Non-Classical Inductive Bias</i>]
    end

    A -->|Limitation: Fails on 3D Rotation| B
    B -->|Limitation: Loses Spatial Coordinates| C
    C -->|Hypothesis: Classical Logic misses Electron Correlations| D

---

## 🧠 One-Line Summary
*A physically equivariant, hybrid quantum–classical graph model for protein structures, evaluated with rigorous baselines and ablations.*
