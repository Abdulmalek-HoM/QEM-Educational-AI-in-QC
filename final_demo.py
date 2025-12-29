import numpy as np
import torch
import torch.nn as nn
from qiskit import transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import utils

# --- 1. RE-DEFINE THE AI MODEL ARCHITECTURE ---
# Must match the training script (Embedding Dim=16)
class QEM_LSTM(nn.Module):
    def __init__(self, vocab_size=7, embedding_dim=16, hidden_size=32):
        super(QEM_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])

# --- 2. THE HYBRID SOLVER API ---
class HybridQEMSolver:
    def __init__(self, ai_model_path='qem_model_2000_samples.pth'):
        # Initialize ZNE Backend
        self.sim = AerSimulator()
        
        # Load AI Model
        self.ai = QEM_LSTM()
        try:
            self.ai.load_state_dict(torch.load(ai_model_path))
            print(f"✅ AI Model loaded from {ai_model_path}")
        except FileNotFoundError:
            print("⚠️ Model file not found! Using untrained weights (expect poor results).")
        self.ai.eval()
        
        self.gate_map = {"pad": 0, "h": 1, "x": 2, "z": 3, "s": 4, "id": 5, "cx": 6}

    def _tokenize(self, qc):
        tokens = []
        for instruction in qc.data:
            name = instruction.operation.name
            tokens.append(self.gate_map.get(name, 0))
        if len(tokens) < 60: tokens += [0]*(60-len(tokens))
        else: tokens = tokens[:60]
        return torch.tensor([tokens], dtype=torch.long)

    def zne_step(self, qc):
        scales = [1.0, 2.0, 3.0]
        results = []
        for s in scales:
            nm = utils.build_custom_noise_model(noise_scale=s)
            sim_s = AerSimulator(noise_model=nm)
            job = sim_s.run(transpile(qc, sim_s), shots=2000)
            counts = job.result().get_counts()
            shots = sum(counts.values())
            probs = {k: v/shots for k,v in counts.items()}
            exp = probs.get('00',0)+probs.get('11',0) - probs.get('01',0)-probs.get('10',0)
            results.append(exp)
            
        coeffs = np.polyfit(scales, results, 2)
        return np.polyval(coeffs, 0), results[0]  # ZNE, Raw

    def solve(self, qc, return_debug=False):
        print("🔵 Hybrid Solver Started...")
        zne_val, raw_val = self.zne_step(qc)
        print(f"   1. Raw Value:  {raw_val:.3f}")
        print(f"   2. ZNE Result: {zne_val:.3f} (Poly Extrapolation)")
        
        with torch.no_grad():
            features = self._tokenize(qc)
            ai_correction = self.ai(features).item()
        
        print(f"   3. AI Offset:  {ai_correction:+.3f}")
        final_val = zne_val + ai_correction
        print(f"🟢 Final Result:  {final_val:.3f}")
        
        if return_debug:
             return final_val, zne_val, raw_val
        return final_val

# --- 3. RUN THE DEMO ---
def main():
    # 1. Create a Challenging Circuit (Depth 40)
    print("Generating Challenge Circuit (Depth 40)...")
    qc = utils.create_random_clifford_circuit(2, 40)
    qc.measure_all()

    # 2. Calculate Ideal Ground Truth
    sim_ideal = AerSimulator(method='stabilizer')
    job_id = sim_ideal.run(transpile(qc, sim_ideal), shots=2000)
    counts_id = job_id.result().get_counts()
    ideal_val = (counts_id.get('00',0)+counts_id.get('11',0)-counts_id.get('01',0)-counts_id.get('10',0))/2000
    print(f"⭐ Ideal Value:    {ideal_val:.3f}")

    # 3. Run Hybrid Solver
    solver = HybridQEMSolver()
    hybrid, zne, raw = solver.solve(qc, return_debug=True)

    # 4. Plot Results
    labels = ['Raw (Noisy)', 'ZNE Only', 'Hybrid (AI+ZNE)', 'Ideal']
    values = [raw, zne, hybrid, ideal_val]
    colors = ['red', 'orange', 'green', 'blue']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors, alpha=0.8)
    plt.title('Error Mitigation Performance (Depth 40)', fontsize=14)
    plt.ylabel('Expectation Value <ZZ>')
    plt.ylim(-1.1, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add values on top
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.05 if v >= 0 else v - 0.1, 
                 f"{v:.2f}", ha='center', fontweight='bold')

    plt.axhline(ideal_val, color='blue', linestyle='--', label='Target')
    plt.legend()
    plt.savefig('final_demo_result.png')
    print("✅ Demo Complete. Plot saved to final_demo_result.png")

if __name__ == "__main__":
    main()
