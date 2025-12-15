# Hybrid AI-ZNE: Data-Driven Quantum Error Mitigation
### AQC "Hack the Horizon" Challenge Submission

## 🚀 Overview
This project implements a "Zero-to-Hero" pipeline for Quantum Error Mitigation (QEM) that combines physics-based **Zero-Noise Extrapolation (ZNE)** with data-driven **Artificial Intelligence (SVR/LSTM)**.

While ZNE effectively mitigates linear noise scaling, it fails when circuit depth exceeds the decoherence threshold ($D > 25$), often flipping the logical state sign. Our hybrid model uses AI to predict the residual non-linear error that ZNE misses, effectively extending the usable depth of the quantum processor.

## 📊 Scientific Insight & Results
**Hypothesis:** Deep Learning models can capture non-Markovian noise accumulation (memory effects) in transmon qubits better than simple linear extrapolation.

**Validation:**
We validated the pipeline on random Clifford circuits with depths up to 30.
* **Baseline (ZNE Only):** Failed to recover the state, yielding an error of **1.498** (sign flip).
* **Hybrid (AI + ZNE):** Corrected the sign and reduced error to **0.826**.
* **Final Improvement Ratio:** **1.81x** (Success > 1.0)

## 🛠️ Installation
**Critical:** This project requires Qiskit 0.46 (Stable) to support the Pulse Physics simulations in Module 1.

```bash
pip install -r requirements.txt
