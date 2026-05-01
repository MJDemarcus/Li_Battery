import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from src.data_preprocessor import preprocess_data
import os

def train_spike_detector():
    # 1. Get Data
    df, _, _ = preprocess_data()
    if df is None:
        return None
    
    # 2. Prepare Features
    features = ['voltage', 'current', 'temperature', 'dT_dt', 'dV_dt']
    X = df[features]
    y = df['label']
    
    # 3. Train Model
    print("Training Random Forest Spike Detector...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 4. Predict and Evaluate
    y_pred = rf.predict(X)
    print("Random Forest Performance:")
    print(classification_report(y, y_pred))
    
    # 5. Visualize Detection
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['temperature'], label='Temperature (C)', color='blue')
    plt.scatter(df['time'][y_pred == 1], df['temperature'][y_pred == 1], 
                color='red', label='RF Detected Runaway', alpha=0.5)
    plt.title('Thermal Runaway Detection (Random Forest)')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (C)')
    plt.legend()
    
    plot_path = 'rf_detection_plot.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    return rf

if __name__ == "__main__":
    train_spike_detector()
