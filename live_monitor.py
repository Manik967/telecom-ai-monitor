import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from data_processor import create_training_data, standardize_data, FEATURE_COLS
from model_handler import train_isolation_forest
import time

# --- CONFIGURATION ---
WINDOW_SIZE = 50  # Number of points shown on screen at once
UPDATE_SPEED = 0.5 # Seconds between "live" data points

# 7x7 Strategy: 7 Features detecting 7 Signatures
SIGNATURES = {
    'DDoS Attack': {'T': 200, 'L': 100, 'E': 15, 'J': 5, 'Th': -70, 'Loss': 6, 'CPU': 10},
    'Fiber Cut': {'T': -100, 'L': 180, 'E': 8, 'J': 10, 'Th': -80, 'Loss': 8, 'CPU': 15},
    'Probe Attack': {'T': 0, 'L': 0, 'E': 45, 'J': 0, 'Th': 0, 'Loss': 0, 'CPU': 0},
    'Hardware Bottleneck': {'T': 100, 'L': 100, 'E': 1, 'J': 3.5, 'Th': -40, 'Loss': 2, 'CPU': 20},
    'Resource Exhaustion': {'T': 120, 'L': 50, 'E': 6, 'J': 2, 'Th': -15, 'Loss': 2, 'CPU': 70},
    'Routing Loop': {'T': 10, 'L': 200, 'E': 1.5, 'J': 12, 'Th': 0, 'Loss': 9, 'CPU': 5},
    'VoIP Degradation': {'T': 0, 'L': 150, 'E': 2, 'J': 20, 'Th': -10, 'Loss': 8, 'CPU': 5}
}

class LiveNetworkMonitor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        # Deques keep the last N points to create a scrolling effect
        self.data_windows = {col: deque(maxlen=WINDOW_SIZE) for col in FEATURE_COLS}
        self.time_window = deque(maxlen=WINDOW_SIZE)
        self.anomaly_window = deque(maxlen=WINDOW_SIZE)
        self.counter = 0
        
        # Injection state
        self.active_attack = None
        self.attack_timer = 0

    def generate_live_point(self):
        """Simulates one interval of live network traffic with noise and potential anomalies."""
        self.counter += 1
        # Baseline normal values
        point = {
            'Traffic_Volume': 110 + 5 * np.random.randn(),
            'Latency_ms': 55 + 5 * np.random.randn(),
            'Error_Rate': 1.2 + 0.3 * np.random.randn(),
            'Jitter_ms': 2.5 + 0.2 * np.random.randn(),
            'Throughput_Mbps': 75 + 4 * np.random.randn(),
            'Packet_Loss_Rate': 0.6 + 0.1 * np.random.randn(),
            'CPU_Utilization_Pct': 35 + 3 * np.random.randn()
        }

        # If an attack is active, apply the signature deviations
        if self.active_attack:
            self.attack_timer -= 1
            params = SIGNATURES[self.active_attack]
            
            for col in FEATURE_COLS:
                # Map feature names to shorthand keys used in SIGNATURES dictionary
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
                    if self.active_attack == 'Fiber Cut' and (col == 'Traffic_Volume' or col == 'Throughput_Mbps'):
                        # Physical layer failure forces near-zero readings
                        point[col] = 1.0 + abs(np.random.randn() * 0.5)
                    else:
                        point[col] += val
                        # Maintain physical reality (cannot be below zero)
                        if point[col] < 0: point[col] = 0.5 + abs(np.random.randn() * 0.2)

            if self.attack_timer <= 0:
                print(f"\n[RESOLVED] {self.active_attack} pattern cleared. System status: NORMAL.")
                self.active_attack = None

        return point

    def start_monitoring(self):
        plt.ion() # Interactive mode for live plotting
        fig, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
        plt.subplots_adjust(hspace=0.5)
        fig.patch.set_facecolor('#0b0f1a') # Dark dashboard theme
        
        print("\n" + "="*50)
        print(">>> LIVE NETWORK MONITORING STARTED")
        print(">>> Injecting one of 7 signatures every 30 intervals.")
        print("="*50)

        try:
            while True:
                # 1. Get new data
                new_point = self.generate_live_point()
                df_point = pd.DataFrame([new_point])
                
                # 2. ML Inference
                scaled_point = self.scaler.transform(df_point[FEATURE_COLS])
                prediction = self.model.predict(scaled_point)[0]
                
                # 3. Update Buffers
                self.time_window.append(self.counter)
                self.anomaly_window.append(prediction)
                for col in FEATURE_COLS:
                    self.data_windows[col].append(new_point[col])

                # 4. Refresh Plots
                for i, col in enumerate(FEATURE_COLS):
                    ax = axes[i]
                    ax.clear()
                    ax.set_facecolor('#1a1a1a')
                    
                    # Plot the line
                    ax.plot(list(self.time_window), list(self.data_windows[col]), color='cyan', alpha=0.8, lw=1.5)
                    
                    # Highlight anomalies in Red
                    times = list(self.time_window)
                    vals = list(self.data_windows[col])
                    anoms = list(self.anomaly_window)
                    
                    anom_x = [times[j] for j in range(len(anoms)) if anoms[j] == -1]
                    anom_y = [vals[j] for j in range(len(anoms)) if anoms[j] == -1]
                    
                    if anom_x:
                        ax.scatter(anom_x, anom_y, color='red', s=40, zorder=5)
                        if i == 0:
                             print(f"![{self.counter}] ALERT: {self.active_attack or 'UNKNOWN'} SIGNATURE DETECTED!", end='\r')

                    ax.set_ylabel(col.split('_')[0], color='white', fontsize=7)
                    ax.tick_params(axis='both', colors='gray', labelsize=7)
                    ax.grid(True, color='#333', alpha=0.2)

                plt.suptitle(f"Continuous 7D Telemetry | Event: {self.active_attack or 'Normal Baseline'}", color='#adff2f', fontsize=12)
                
                plt.pause(UPDATE_SPEED)
                
                # Random Injection Logic
                if self.counter % 30 == 0 and not self.active_attack:
                    choice = np.random.choice(list(SIGNATURES.keys()))
                    print(f"\n[SYSTEM] Injecting simulated {choice} pattern...")
                    self.active_attack = choice
                    self.attack_timer = 12

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    # Setup
    print("Initializing Engine (Fitting baseline to training cluster)...")
    # Training data must include the signatures we want to isolate
    df_train = create_training_data()
    scaler = standardize_data(df_train)
    model = train_isolation_forest(df_train, scaler)
    
    monitor = LiveNetworkMonitor(model, scaler)
    monitor.start_monitoring()