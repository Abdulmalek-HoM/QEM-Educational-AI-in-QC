import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from qiskit_aer import AerSimulator
from qiskit import transpile
import utils

# --- HYPERPARAMETERS ---
DATASET_SIZE = 2000  # Scaled up for better generalization
MAX_SEQ_LEN = 60
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "qem_model_2000_samples.pth"

# --- 1. TOKENIZER ---
class CircuitTokenizer:
    def __init__(self):
        self.gate_map = {"pad": 0, "h": 1, "x": 2, "z": 3, "s": 4, "id": 5, "cx": 6}
        self.vocab_size = len(self.gate_map)

    def tokenize(self, instruction_list):
        return [self.gate_map.get(g, 0) for g in instruction_list]

    def pad_sequence(self, tokenized_seq, max_len):
        if len(tokenized_seq) >= max_len:
            return tokenized_seq[:max_len]
        return tokenized_seq + [0] * (max_len - len(tokenized_seq))

# --- 2. MODEL ---
class QEM_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_size=32):
        super(QEM_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        last_hidden_state = h_n[-1]
        prediction = self.fc(last_hidden_state)
        return prediction

def main():
    print(f"🚀 Starting QEM Training Pipeline")
    print(f"Generating {DATASET_SIZE} circuits...")

    tokenizer = CircuitTokenizer()
    
    X_sequences = []
    y_targets = []
    
    sim_ideal = AerSimulator(method='stabilizer')
    # Using the shared noise model from utils
    noise_model = utils.build_custom_noise_model()
    sim_noisy = AerSimulator(noise_model=noise_model)

    for i in range(DATASET_SIZE):
        try:
            # 1. Generate Random Circuit
            depth = random.randint(5, 50)
            qc, instructions = utils.create_random_clifford_circuit(2, depth, return_instructions=True)
            
            # 2. Tokenize
            tokens = tokenizer.tokenize(instructions)
            padded_tokens = tokenizer.pad_sequence(tokens, MAX_SEQ_LEN)
            X_sequences.append(padded_tokens)
            
            # 3. Simulate
            qc.measure_all()
            qc_t = transpile(qc, sim_noisy) # Transpile for checking, but we simulate ideal/noisy
            
            # Ideal
            res_ideal = sim_ideal.run(qc_t, shots=1000).result().get_counts()
            exp_ideal = (res_ideal.get('00', 0) + res_ideal.get('11', 0) - res_ideal.get('01', 0) - res_ideal.get('10', 0)) / 1000
            
            # Noisy
            res_noisy = sim_noisy.run(qc_t, shots=1000).result().get_counts()
            exp_noisy = (res_noisy.get('00', 0) + res_noisy.get('11', 0) - res_noisy.get('01', 0) - res_noisy.get('10', 0)) / 1000
            
            # Target = Error
            y_targets.append(exp_ideal - exp_noisy)
            
            if (i+1) % 100 == 0:
                print(f"  Generated {i+1}/{DATASET_SIZE}...")
                
        except Exception as e:
            print(f"Skipping circuit {i} due to error: {e}")

    # Convert to Tensor
    X_tensor = torch.tensor(X_sequences, dtype=torch.long)
    y_tensor = torch.tensor(y_targets, dtype=torch.float32).unsqueeze(1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2)
    print("✅ Dataset Ready.")

    # Training
    model = QEM_LSTM(vocab_size=tokenizer.vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Starting Training...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | MSE Loss: {epoch_loss / len(train_loader):.5f}")

    # Save
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")
    
    # Verification on Test Set
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_loss = criterion(test_preds, y_test)
        print(f"📉 Final Test MSE: {test_loss.item():.5f}")

if __name__ == "__main__":
    main()
