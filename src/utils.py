import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length):
    """
    Creates sequences for LSTM model.
    
    Args:
        data: Numpy array of data.
        seq_length: Length of sequences.
        
    Returns:
        np.array: X sequences.
        np.array: y targets.
    """
    # Vectorized implementation for speed
    xs = []
    ys = []
    
    # Ensure data is numpy array
    data = np.array(data)
    
    # Use simple loop if data is small, but for larger data this is effectively efficient enough
    # compared to the LSTM training time. 
    # For a true vectorized approach we would use stride_tricks, but that can be complex to safeguard.
    # Let's stick to a list comprehension which is faster than append loop
    xs = [data[i:(i + seq_length)] for i in range(len(data) - seq_length)]
    ys = [data[i + seq_length] for i in range(len(data) - seq_length)]
    
    return np.array(xs), np.array(ys)

def preprocess_data(df, target_col, seq_length=50, split_ratio=0.8):
    """
    Preprocesses data for LSTM model: scaling and sequence creation.
    
    Args:
        df: Pandas DataFrame.
        target_col: Name of the target column.
        seq_length: Length of input sequences.
        split_ratio: Train/test split ratio.
        
    Returns:
        dict: Contains X_train, y_train, X_test, y_test, scaler.
    """
    data = df[[target_col]].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = create_sequences(data_scaled, seq_length)
    
    train_size = int(len(X) * split_ratio)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler
    }
