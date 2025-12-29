
import random
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, ReadoutError

# ==========================================
# 1. SHARED NOISE MODEL (Single Source of Truth)
# ==========================================
def build_custom_noise_model(t1=50e-6, t2=70e-6, 
                           time_1q=50e-9, time_2q=400e-9, 
                           readout_err=0.05, 
                           noise_scale=1.0):
    """
    Constructs a noise model mimicking a real IBM device.
    Supports scaling for ZNE (noise_scale > 1.0 means worse noise).

    Args:
        t1, t2: Relaxation and Dephasing times (seconds)
        time_1q: Duration of single-qubit gates (seconds)
        time_2q: Duration of two-qubit gates (seconds)
        readout_err: Probability of measurement flip
        noise_scale: Multiplier for error rates (1.0 = baseline, 2.0 = double noise)
    """
    # Scale T1/T2 (Higher scale = Faster decay = Smaller T1/T2)
    s_t1 = t1 / noise_scale
    s_t2 = t2 / noise_scale
    
    # Scale Readout (Higher scale = More flips)
    s_readout = min(0.5, readout_err * noise_scale)

    noise_model = NoiseModel()

    # --- 1. Thermal Relaxation Errors ---
    # 1-Qubit Gates
    error_1q = thermal_relaxation_error(s_t1, s_t2, time_1q)
    noise_model.add_all_qubit_quantum_error(error_1q, ['sx', 'x', 'id', 'rz', 'h'])

    # 2-Qubit Gates (Tensor Product)
    error_2q_single = thermal_relaxation_error(s_t1, s_t2, time_2q)
    error_2q = error_2q_single.expand(error_2q_single) 
    noise_model.add_all_qubit_quantum_error(error_2q, ['ecr', 'cx'])

    # --- 2. Readout Error ---
    probabilities = [
        [1 - s_readout, s_readout],  # P(0|0), P(1|0)
        [s_readout, 1 - s_readout]   # P(0|1), P(1|1)
    ]
    readout_error = ReadoutError(probabilities)
    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model

# ==========================================
# 2. RANDOM CIRCUIT GENERATOR
# ==========================================
def create_random_clifford_circuit(num_qubits, depth, return_instructions=False):
    """
    Generates a random circuit using only Clifford gates.
    
    Args:
        num_qubits: Number of qubits
        depth: Number of layers
        return_instructions: If True, returns (circuit, instruction_list)
                             instruction_list is ['h', 'cx', 'z', ...] useful for Tokenization
    """
    qc = QuantumCircuit(num_qubits)
    gates_1q = ['h', 'x', 'z', 'id', 's']
    instruction_list = []

    for _ in range(depth):
        q = random.randint(0, num_qubits - 1)
        
        # 50% chance of 2-qubit gate (if possible)
        if num_qubits > 1 and random.random() > 0.5:
            target = random.randint(0, num_qubits - 1)
            while target == q:
                 target = random.randint(0, num_qubits - 1)
            qc.cx(q, target)
            instruction_list.append('cx')
        else:
            g = random.choice(gates_1q)
            getattr(qc, g)(q)
            instruction_list.append(g)
            
    if return_instructions:
        return qc, instruction_list
    return qc
