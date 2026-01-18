# Quantum Equivariant GNNs for Protein Modeling

**A hybrid quantum–classical protein modeling project** combining *SE(3)-equivariant Graph Neural Networks (EGNNs)* with a *Variational Quantum Circuit (VQC)* to study how quantum-induced inductive bias affects learning behavior in physically grounded models.

---

## 🔍 What This Project Does
* **Represents proteins** as 3D graphs.
* **Uses SE(3)-equivariant GNNs** to respect physical symmetries (rotation, translation).
* **Adds a quantum layer** as a learnable feature transformation.
* **Compares classical vs. quantum heads** under matched conditions.
* **Focuses on behavior & inductive bias**, *not* quantum speedup.

---

## ⚛️ Why a Quantum Layer?
> *A deeper MLP is not equivalent to a quantum circuit.*

| Feature | Deeper MLP | Variational Quantum Circuit (VQC) |
| :--- | :--- | :--- |
| **Non-linearity** | Standard classical (ReLU/Sigmoid) | **Trigonometric & interference-based** |
| **Inductive Bias** | High capacity, more of the same | **Different bias (entanglement, global correlations)** |
| **Learning Style** | Memorizes quickly | **More constrained; acts as an implicit regularizer** |

**The goal is not "better performance," but *different* learning behavior.**

---

## 🧪 Why the Ablation (Step 6)?
Without ablations, the project would simply be: *"I added a quantum circuit and it trained."* **That’s not research.**

Instead, we ask: 
*Does a quantum-induced inductive bias behave differently from a classical one under matched conditions?*

* ❌ **Not better.**
* ❌ **Not faster.**
* ✅ **Just different.**

---

## 📊 Key Observation
* **Classical head** $\rightarrow$ *Fast memorization, large oscillations.*
* **Quantum head** $\rightarrow$ *Slower, smoother convergence.*

> **Takeaway:** > **Classical** = Fast Fit  
> **Quantum** = Implicit Regularization

---

## 🧱 What’s Implemented
1.  **PDB $\rightarrow$ 3D protein graphs** processing pipeline.
2.  **SE(3)-equivariant EGNN** backbone.
3.  **Classical MLP head** (Baseline).
4.  **Quantum (VQC) head** (Experimental).
5.  **Controlled comparison** + visualization tools.

---

## ⚠️ Scope
* **Quantum simulation only** (no hardware execution).
* **Single-protein experiments** for behavioral analysis.
* **No claims of quantum advantage** (focus is on correctness).

---

## 🧠 One-Line Summary
*A physically equivariant, hybrid quantum–classical graph model for protein structures, evaluated with rigorous baselines and ablations.*
