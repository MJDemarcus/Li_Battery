import os
# Disable GPU/Metal for fast CPU training of tiny model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_preprocessor import preprocess_data
from src.spike_detector_rf import train_spike_detector
from src.trend_analyzer_lstm import build_cnn_lstm
import tensorflow as tf

# Force CPU only
tf.config.set_visible_devices([], 'GPU')

def run_early_warning_system():
    print("Initializing Multi-Model Early Warning System...")
    
    # 1. Train/Get RF Model
    rf = train_spike_detector()
    
    # 2. Get Data for LSTM
    window_size = 10
    df, X, y = preprocess_data(window_size=window_size)
    
    # 3. Build and Train LSTM
    input_shape = (X.shape[1], X.shape[2])
    print("Initializing Integrated CNN-LSTM...")
    lstm_model = build_cnn_lstm(input_shape)
    print("Training Integrated CNN-LSTM (1 epoch, CPU only)...")
    lstm_model.fit(X, y, epochs=1, batch_size=4, verbose=0)
    
    # 4. Comparative Inference
    features = ['voltage', 'current', 'temperature', 'dT_dt', 'dV_dt']
    rf_preds = rf.predict(df[features])
    
    lstm_preds_raw = lstm_model.predict(X)
    lstm_preds = (lstm_preds_raw > 0.5).astype(int).flatten()
    
    # 5. Visualization
    plt.figure(figsize=(12, 7))
    plt.plot(df['time'], df['temperature'], label='Temperature (C)', color='gray', alpha=0.5)
    
    # Highlight RF Detections
    rf_times = df['time'][rf_preds == 1]
    rf_temps = df['temperature'][rf_preds == 1]
    plt.scatter(rf_times, rf_temps, color='red', marker='x', label='RF Spike Detection')
    
    # Highlight LSTM Detections
    lstm_times = df['time'].iloc[window_size:][lstm_preds == 1]
    lstm_temps = df['temperature'].iloc[window_size:][lstm_preds == 1]
    plt.scatter(lstm_times, lstm_temps, color='orange', marker='o', facecolors='none', 
                s=100, label='LSTM Trend Warning')
    
    plt.axhline(80, color='red', linestyle='--', alpha=0.3, label='Runaway Threshold')
    plt.title('Integrated Early Warning System: Detection Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (C)')
    plt.legend()
    
    final_plot = 'early_warning_detection.png'
    plt.savefig(final_plot)
    print(f"Final detection comparison saved to {final_plot}")
    
    # 6. Report
    onset_idx = df[df['label'] == 1].index[0]
    onset_time = df['time'].iloc[onset_idx]
    first_rf_time = rf_times.iloc[0] if not rf_times.empty else None
    first_lstm_time = lstm_times.iloc[0] if not lstm_times.empty else None
    
    print(f"\n" + "="*45)
    print(f"--- THERMAL RUNAWAY DIAGNOSTIC REPORT ---")
    print(f"="*45)
    print(f"Critical Runaway Onset (Phase 2):   {onset_time:.2f}s\n")
    
    if first_rf_time: 
        rf_lead_time = onset_time - first_rf_time
        print(f"[RF Detector] Warning Triggered at: {first_rf_time:.2f}s")
        print(f"              => Lead Time:         {rf_lead_time:.2f}s before failure\n")
        
    if first_lstm_time: 
        lstm_lead_time = onset_time - first_lstm_time
        print(f"[CNN-LSTM]    Warning Triggered at: {first_lstm_time:.2f}s")
        print(f"              => Lead Time:         {lstm_lead_time:.2f}s before failure")
    print(f"="*45 + "\n")

if __name__ == "__main__":
    run_early_warning_system()
