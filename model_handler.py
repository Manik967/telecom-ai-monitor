from sklearn.ensemble import IsolationForest
from data_processor import FEATURE_COLS

def train_isolation_forest(df_train, scaler):
    """Trains the Isolation Forest model on standardized training data."""
    X_train = df_train[FEATURE_COLS].copy()
    X_scaled = scaler.transform(X_train) 
    
    # We define contamination as a small percentage of expected anomalies
    model = IsolationForest(
        n_estimators=100,
        contamination=0.015,
        random_state=42
    )

    model.fit(X_scaled)
    print("Step 3 (Training) complete. Model trained on 7D space.")
    return model