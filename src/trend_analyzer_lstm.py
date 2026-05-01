import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, TimeDistributed, MaxPooling1D
import matplotlib.pyplot as plt
from src.data_preprocessor import preprocess_data

def build_cnn_lstm(input_shape):
    model = Sequential([
        # CNN stage to extract spatial features from subsequences
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        
        # LSTM stage to capture temporal dependencies
        LSTM(50, return_sequences=False),
        
        # Dense layers for final classification/regression
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid') # Classification: Is runaway onset imminent?
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_trend_analyzer():
    # 1. Get Preprocessed Sequential Data
    window_size = 10
    df, X, y = preprocess_data(window_size=window_size)
    if X is None:
        return None
    
    # 2. Build Model
    # Input shape: (window_size, num_features)
    input_shape = (X.shape[1], X.shape[2])
    model = build_cnn_lstm(input_shape)
    model.summary()
    
    # 3. Train
    print("Training CNN-LSTM Trend Analyzer...")
    # Training for more epochs since dataset is tiny
    history = model.fit(X, y, epochs=50, batch_size=4, verbose=0)
    
    # 4. Predict
    y_pred = (model.predict(X) > 0.5).astype(int)
    
    # 5. Visualize Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(y, label='Actual Label', marker='o', alpha=0.5)
    plt.plot(y_pred, label='CNN-LSTM Predicted', marker='x', alpha=0.7)
    plt.title('Thermal Runaway Detection (CNN-LSTM)')
    plt.xlabel('Sequence Index')
    plt.ylabel('Is Runaway?')
    plt.legend()
    
    plot_path = 'cnn_lstm_detection_plot.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    return model

if __name__ == "__main__":
    train_trend_analyzer()
