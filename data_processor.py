import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Defines the 7 features used across all modules
FEATURE_COLS = [
    'Traffic_Volume', 'Latency_ms', 'Error_Rate', 'Jitter_ms', 
    'Throughput_Mbps', 'Packet_Loss_Rate', 'CPU_Utilization_Pct'
]

def create_training_data(n_samples=1500, seed=42):
    """
    Creates the synthetic training DataFrame with 7 features and 
    7 distinct anomaly signatures (7x7 Configuration).
    """
    np.random.seed(seed)
    time_index = pd.date_range(start='2025-01-01', periods=n_samples, freq='min')

    # Base Data Generation (7 Features)
    base_traffic = 100 + 10 * np.sin(np.linspace(0, 50, n_samples)) + 5 * np.random.randn(n_samples)
    base_latency = 50 + 5 * np.random.randn(n_samples) + 2 * np.sin(np.linspace(0, 50, n_samples))
    base_errors = 1 + 0.5 * np.random.randn(n_samples)
    base_jitter = 2 + 0.2 * np.random.randn(n_samples) 
    base_throughput = 80 + 5 * np.random.randn(n_samples)
    base_loss = 0.5 + 0.1 * np.random.randn(n_samples)
    base_cpu = 30 + 5 * np.random.randn(n_samples)

    df = pd.DataFrame({
        'Timestamp': time_index,
        'Traffic_Volume': base_traffic, 
        'Latency_ms': base_latency, 
        'Error_Rate': base_errors,
        'Jitter_ms': base_jitter, 
        'Throughput_Mbps': base_throughput,
        'Packet_Loss_Rate': base_loss, 
        'CPU_Utilization_Pct': base_cpu
    })

    # --- Inject 7 Anomaly Signatures for Training ---
    
    # 1. DDoS (High volume/latency flood)
    df.loc[150:155, FEATURE_COLS] += np.array([150, 70, 10, 5.0, -60, 4.0, 5])
    
    # 2. VoIP Degradation (High Jitter/Loss - Added for 7x7 alignment)
    df.loc[250:255, ['Latency_ms', 'Jitter_ms', 'Packet_Loss_Rate']] += np.array([150, 20.0, 8.0])
    
    # 3. Fiber Cut (Near Zero Traffic/Throughput)
    df.loc[400:405, ['Traffic_Volume', 'Throughput_Mbps']] = 1 
    df.loc[400:405, ['Latency_ms', 'Error_Rate', 'Jitter_ms', 'Packet_Loss_Rate', 'CPU_Utilization_Pct']] += np.array([150, 8, 8.0, 6.0, 10])
    
    # 4. Probe Attack (Security Scan)
    df.loc[650:655, 'Error_Rate'] += 40
    
    # 5. Hardware Bottleneck (Congestion)
    df.loc[900:905, ['Traffic_Volume', 'Latency_ms', 'Jitter_ms', 'Throughput_Mbps']] += np.array([80, 90, 3.5, -30])
    
    # 6. Resource Exhaustion (CPU Spike)
    df.loc[1150:1155, ['Traffic_Volume', 'CPU_Utilization_Pct']] += np.array([120, 60])
    
    # 7. Routing Loop (Latency/Jitter/Loss loop)
    df.loc[1350:1355, ['Latency_ms', 'Jitter_ms', 'Packet_Loss_Rate']] += np.array([180, 10.0, 7.0])

    print("Step 1: Training data generated with 7 anomaly signatures.")
    return df

def standardize_data(df):
    """Fits scaler on training data and returns the scaler object."""
    X = df[FEATURE_COLS].copy()
    scaler = StandardScaler()
    scaler.fit(X)
    print("Step 2: Scaler fitted on 7 features.")
    return scaler