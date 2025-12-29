# 🚀 Team Attack Plan: AQC Hack The Horizon
**Project:** QEM-Educational-AI-in-QC (Hybrid AI-ZNE Error Mitigation)

## 🎯 Mission
Win the "Hack the Horizon" challenge by demonstrating a scientifically valid, educationally rich, and working prototype of AI-driven Quantum Error Mitigation.

## 👥 Roles & Responsibilities

### 1. ⚛️ Quantum Physicist (Scientific Lead)
**Focus:** Physics correctness, ZNE implementation, Noise Modeling.
*   [ ] **Audit Module 4 (Noise):** Ensure the custom noise model in `utils.py` accurately reflects real hardware ($T_1, T_2$ params).
*   [ ] **Refine ZNE (Module 5):** Verify polynomial/exponential extrapolation logic. Ensure it's clearly explained.
*   [ ] **Benchmarking:** Generate purely random Clifford circuits (Depth 5-50) to prove ZNE failure points (show the "sign flips").
*   [ ] **Documentation:** Write the "Scientific Insight" sections in notebooks. Explain *why* ZNE fails.

### 2. 🧠 AI Engineer (ML Lead)
**Focus:** LSTM Model, Data generation, Training pipeline.
*   [ ] **Scale Dataset:** Run `Module_3_Large_Scale.ipynb` (or the equivalent in Module 6) to generate ~2000+ samples.
*   [ ] **Model Tuning:** Experiment with LSTM hyperparameters (Layers, Hidden Size) to maximize "Error Reduction Factor".
*   [ ] **Generalization Test:** Ensure the model works on *unseen* circuit structures, not just the training distribution.
*   [ ] **Save Model:** Export the trained PyTorch model (`.pth`) so it can be loaded in Module 7 without retraining.

### 3. 🛠️ Integration & Deployment (DevOps/Full Stack)
**Focus:** Repository stability, Dependencies, Final Demo.
*   [ ] **Dependency Lock:** Freeze `requirements.txt` (ensure Qiskit 0.46 compatibility).
*   [ ] **Utils Refactor:** Ensure `utils.py` is robust and documented. Move common plotting code there.
*   [ ] **Module 7 (Deployment):** Build the "Grand Finale".
    *   Load the saved LSTM model.
    *   Accept a *user-provided* Qiskit circuit.
    *   Output: `Original Error` vs `Mitigated Error` graph.
    *   (Optional) Wrap this in a simple `Streamlit` app for a web UI.

### 4. 🎨 Creative & Storyteller (Presentation Lead)
**Focus:** "Selling" the project, Video, README, Educational Value.
*   [ ] **Notebook Polish:** Go through Modules 1-7. Add Emojis, clear Headers, and simpler explanations. Make it "readable like a book".
*   [ ] **The Hook:** Rewrite the README introduction to be punchy (Problem -> Solution -> Impact).
*   [ ] **Video Asset:** Record the 3-minute pitch.
    *   0:00-0:30: The Problem (Quantum Noise kills apps).
    *   0:30-1:00: The Solution (Hybrid AI+ZNE).
    *   1:00-2:00: Determining the Tech (Show the LSTM learning).
    *   2:00-2:30: Live Demo (Module 7).
    *   2:30-3:00: Impact/Future.

---

## 📅 Execution Roadmap

### Phase 1: Solidify (Days 1-2)
*   **Tech:** Fix any bugs in `utils.py`. Generate the "Golden Dataset" (>=2000 samples).
*   **Science:** Prove ZNE vs. AI diff (get that ">1 improvement" chart solid).

### Phase 2: Polish (Day 3)
*   **Education:** Add markdown explanations to all cells, when needed."
*   **UI:** Make the charts pretty (Matplotlib themes, clear legends).

### Phase 3: Ship (Final Day)
*   **Video:** Record and edit.
*   **Submission:** Check all hackathon checkboxes (License, Open Source, Links).
