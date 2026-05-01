import pandas as pd
import numpy as np
import os

def preprocess_data(input_path='data/thermal_runaway_data.csv', window_size=5):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return None, None, None

    df = pd.read_csv(input_path)
    # Remove rows with comments if any (though standard CSV doesn't have them, the user's sample did)
    df = df[pd.to_numeric(df['time'], errors='coerce').notnull()].copy()
    df = df.apply(pd.to_numeric)

    # 1. Feature Engineering: Rates of change
    # Since 'time' intervals are not always 1, we divide by time diff
    df['dt'] = df['time'].diff().fillna(1)
    df['dT_dt'] = df['temperature'].diff().fillna(0) / df['dt']
    df['dV_dt'] = df['voltage'].diff().fillna(0) / df['dt']
    
    # 2. Labeling: Runaway onset
    # We define runaway as when Temperature > 80 OR dT_dt > 1.0 (arbitrary for sample)
    df['label'] = ((df['temperature'] > 80) | (df['dT_dt'] > 1.0)).astype(int)

    # 3. Sequencing for CNN-LSTM
    features = ['voltage', 'current', 'temperature', 'dT_dt', 'dV_dt']
    X_seq = []
    y_seq = []
    
    data_array = df[features].values
    labels_array = df['label'].values
    
    for i in range(len(data_array) - window_size):
        X_seq.append(data_array[i:i+window_size])
        y_seq.append(labels_array[i+window_size])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"Preprocessed {len(df)} rows. Created {len(X_seq)} sequences.")
    return df, X_seq, y_seq

if __name__ == "__main__":
    df, X, y = preprocess_data()
    if df is not None:
        print("Features sample:")
        print(df[['time', 'temperature', 'dT_dt', 'label']].tail(15))
