import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processor import create_training_data, standardize_data, FEATURE_COLS
from model_handler import train_isolation_forest

# Define dynamic signatures for the validation phase

SIGNATURES = {
    'DDoS Attack': {'T': 200, 'L': 100, 'E': 15, 'J': 5, 'Th': -70, 'Loss': 6, 'CPU': 10, 'desc': 'High traffic and latency flood (Security)'},
    'Fiber Cut': {'T': -100, 'L': 180, 'E': 8, 'J': 10, 'Th': -80, 'Loss': 8, 'CPU': 15, 'desc': 'Total throughput loss (Physical)'},
    'Probe Attack': {'T': 0, 'L': 0, 'E': 45, 'J': 0, 'Th': 0, 'Loss': 0, 'CPU': 0, 'desc': 'Spike in error rates (Security)'},
    'Hardware Bottleneck': {'T': 100, 'L': 100, 'E': 1, 'J': 3.5, 'Th': -40, 'Loss': 2, 'CPU': 20, 'desc': 'Resource congestion (Performance)'},
    'Resource Exhaustion': {'T': 120, 'L': 50, 'E': 6, 'J': 2, 'Th': -15, 'Loss': 2, 'CPU': 70, 'desc': 'Critical CPU utilization (Security/Performance)'},
    'Routing Loop': {'T': 10, 'L': 200, 'E': 1.5, 'J': 12, 'Th': 0, 'Loss': 9, 'CPU': 5, 'desc': 'Infinite packet loops (Operations)'},
    'VoIP Degradation': {'T': 0, 'L': 150, 'E': 2, 'J': 20, 'Th': -10, 'Loss': 8, 'CPU': 5, 'desc': 'Service Quality: High Jitter and Packet Loss (Media/Voice)'}
}

def generate_validation_data(signature_name):
    """
    Creates a fresh batch of unseen test data and injects a single 
    anomaly signature to verify the model's detection capabilities.
    """
    n_samples = 500
    time_index = pd.date_range(start='2026-01-01', periods=n_samples, freq='min')
    
    # Generate random baseline data
    np.random.seed(None) # Truly random on every run
    df_test = pd.DataFrame({
        'Timestamp': time_index,
        'Traffic_Volume': 110 + 5 * np.random.randn(n_samples),
        'Latency_ms': 55 + 5 * np.random.randn(n_samples),
        'Error_Rate': 1.2 + 0.4 * np.random.randn(n_samples),
        'Jitter_ms': 2.5 + 0.3 * np.random.randn(n_samples),
        'Throughput_Mbps': 75 + 4 * np.random.randn(n_samples),
        'Packet_Loss_Rate': 0.6 + 0.2 * np.random.randn(n_samples),
        'CPU_Utilization_Pct': 35 + 4 * np.random.randn(n_samples)
    })

    # Pick a random starting point for the anomaly injection
    start_idx = np.random.randint(150, 350)
    end_idx = start_idx + 4
    params = SIGNATURES[signature_name]

    for col in FEATURE_COLS:
        # Map feature names to shorthand keys (T, L, E, J, Th, Loss, CPU)
        if col.startswith('Traffic'): key = 'T'
        elif col.startswith('Latency'): key = 'L'
        elif col.startswith('Error'): key = 'E'
        elif col.startswith('Jitter'): key = 'J'
        elif col.startswith('Throughput'): key = 'Th'
        elif col.startswith('Packet_Loss'): key = 'Loss'
        elif col.startswith('CPU'): key = 'CPU'
        else: continue
        
        val = params.get(key)
        if val is not None:
            if signature_name == 'Fiber Cut' and (col == 'Traffic_Volume' or col == 'Throughput_Mbps'):
                # For a Fiber Cut, force volume and throughput to near-zero
                df_test.loc[start_idx:end_idx, col] = 1 
            else:
                # Add or subtract the signature deviation
                df_test.loc[start_idx:end_idx, col] += val
                # Clip values to ensure rates/utilization don't go below zero
                df_test[col] = df_test[col].clip(lower=0)

    return df_test

def main():
    print("--- Telecom Anomaly Detection System (7x7 Configuration) ---")
    
    # 1. Initialize data and scaling
    df_train = create_training_data()
    scaler = standardize_data(df_train)
    
    # 2. Train the machine learning model
    model = train_isolation_forest(df_train, scaler)
    
    # 3. Dynamic Validation Test
    choice = np.random.choice(list(SIGNATURES.keys()))
    print(f"\n--- Injecting Unseen Test Case: {choice} ---")
    df_test = generate_validation_data(choice)
    
    # Predict using the trained model and fitted scaler
    X_test_scaled = scaler.transform(df_test[FEATURE_COLS])
    df_test['Anomaly_Flag'] = model.predict(X_test_scaled)
    anomalies = df_test[df_test['Anomaly_Flag'] == -1]

    print(f"Results: Detected {len(anomalies)} anomalous data points.")
    print(f"Interpretation: {SIGNATURES[choice]['desc']}")

    # 4. Visualization of the 7 Metric Panels
    fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    metric_colors = {
        'Traffic_Volume': ('Traffic (GB/min)', '#1e3a8a'),
        'Latency_ms': ('Latency (ms)', 'orange'),
        'Error_Rate': ('Errors (Events/min)', 'red'),
        'Jitter_ms': ('Jitter (ms)', 'purple'),
        'Throughput_Mbps': ('Throughput (Mbps)', 'green'),
        'Packet_Loss_Rate': ('Packet Loss (%)', 'brown'), 
        'CPU_Utilization_Pct': ('CPU Utilization (%)', 'cyan') 
    }

    for i, (col, (label, color)) in enumerate(metric_colors.items()):
        ax = axes[i]
        # Plot the normal baseline flow
        ax.plot(df_test['Timestamp'], df_test[col], color=color, alpha=0.6, label='Flow')
        
        # Highlight detected anomalies with high-visibility markers
        ax.scatter(anomalies['Timestamp'], anomalies[col], 
                    color='#00FF00', s=40, edgecolor='black', zorder=5, label='Anomaly')
        
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.2)
    
    plt.suptitle(f'7D Anomaly Detection Validation: {choice}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    print("\n--- Displaying Visualization Window ---")
    plt.show()

if __name__ == "__main__":
    main()