import numpy as np
import pandas as pd

def detect_anomalies_iqr(df, column, threshold=1.5):
    """
    Detects anomalies using the Interquartile Range (IQR) method.
    
    Args:
        df: Pandas DataFrame.
        column: Column name to check for anomalies.
        threshold: IQR multiplier for outlier detection.
        
    Returns:
        pd.DataFrame: DataFrame with anomalies.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return anomalies

def detect_anomalies_lstm(model, X_data, y_true, scaler, threshold_percentile=95, y_pred=None):
    """
    Detects anomalies based on reconstruction/prediction error.
    """
    if y_pred is None:
        y_pred = model.predict(X_data, verbose=0) # Ensure verbose=0
    
    # Calculate Mean Absolute Error (MAE) loss
    mae_loss = np.mean(np.abs(y_pred - y_true.reshape(-1, 1)), axis=1)
    
    threshold = np.percentile(mae_loss, threshold_percentile)
    
    anomalies_idx = np.where(mae_loss > threshold)[0]
    
    return anomalies_idx, threshold, mae_loss
