import pandas as pd
import numpy as np

def run_hybrid_heuristic(df, target_col='temperature', k_spike=5.0):
    """
    Runs the hybrid heuristic:
    A: Prophet Expected Range Baseline
    B: Derivative Density (Variance/changes in first derivative)
    C: Exponential Spike on 2nd derivative.
    
    Returns annotated DataFrame with sub-scores and probabilities.
    """
    df = df.copy()
    
    # Ensure time is numeric, coercing comment strings
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    df = df.sort_values('time').reset_index(drop=True)
    dt = df['time'].diff().fillna(1.0)
    dt[dt == 0] = 1.0 # prevent division by zero
    
    T = df[target_col]
    
    # Filter high frequency noise (like SNL LCO dt=0.1) by averaging 5 seconds of local data
    T_smooth = T.rolling(window=10, min_periods=1, center=False).mean()
    
    # 1st and 2nd derivatives, also aggressively smoothed
    df['dT_dt'] = T_smooth.diff().fillna(0.0) / dt
    df['dT_dt'] = df['dT_dt'].rolling(window=10, min_periods=1, center=False).mean()
    df['d2T_dt2'] = df['dT_dt'].diff().fillna(0.0) / dt
    df['d2T_dt2'] = df['d2T_dt2'].rolling(window=10, min_periods=1, center=False).mean()
    
    # --- Component A: Causal Statistical Envelope (Replaces Prophet) ---
    rolling_mean = T_smooth.expanding(min_periods=20).mean()
    rolling_std = T_smooth.expanding(min_periods=20).std().fillna(0)
    allowed_upper = rolling_mean + np.maximum(3 * rolling_std, 1.5)
    df['trigger_A'] = (T > allowed_upper).astype(int).cummax()
    
    # --- Component B: Derivative Density ---
    df['heating_flag'] = (df['dT_dt'] > 0.5).astype(int)
    df['trigger_B'] = (df['heating_flag'].rolling(window=10, min_periods=1).mean() >= 0.5).astype(int).cummax()
    
    # --- Component C: 2nd Derivative Spike ---
    df['trigger_C'] = (df['d2T_dt2'] * 100 > 10.0).astype(int).cummax() # equivalent logic adjusted for smooth
    
    # --- Component D: Physics-Informed Digital Twin ---
    T_amb = T.iloc[:50].min()
    n_early = max(50, int(len(df) * 0.25))
    early_df = df.iloc[:n_early]
    
    X = early_df[target_col].values - T_amb
    rate = early_df['dT_dt'].fillna(0.0).values
    
    if len(X) > 1 and len(rate) > 1:
        coeffs = np.polyfit(X, rate, 1)
        beta, alpha = -coeffs[0], coeffs[1]
    else:
        alpha, beta = 0.0, 0.0
        
    X_full = T.values - T_amb
    pred_rate = alpha - beta * X_full
    actual_rate = df['dT_dt'].fillna(0.0).values
    deviation = np.abs(actual_rate - pred_rate)
    
    std_res = np.std(deviation[:n_early]) + 1e-6
    
    # Dev sigma tuned to 2.5. Prevent Trigger D during the baseline fitting window.
    is_anomaly = deviation > 2.5 * std_res
    is_anomaly[:n_early] = False 
    df['trigger_D'] = pd.Series(is_anomaly.astype(int)).cummax()
    
    # --- Discrete State Machine ---
    df['trigger_sum'] = df['trigger_A'] + df['trigger_B'] + df['trigger_C'] + df['trigger_D']
    df['prob_tr'] = (df['trigger_sum'] / 4.0) * 100.0
    
    # Add textual state for output
    df['state_msg'] = np.select(
        [df['trigger_sum'] == 4, df['trigger_sum'] == 3, df['trigger_sum'] == 2, df['trigger_sum'] == 1],
        ['Terminal', 'Critical Risk', 'Unstable (Action)', 'Degraded (Investigate)'],
        default='Healthy'
    )
    
    return df


def get_warning_metrics(df):
    """
    Identifies the exact warning trigger and lead time.
    """
    critical_threshold = 85.0 # 85% probability triggers critical warning
    tr_temp_threshold = 85.0  # Definition of physical TR onset 
    
    warning_idx = df[df['prob_tr'] >= critical_threshold].index
    runaway_idx = df[df['temperature'] >= tr_temp_threshold].index
    
    warning_time = df['time'].iloc[warning_idx[0]] if len(warning_idx) > 0 else None
    runaway_time = df['time'].iloc[runaway_idx[0]] if len(runaway_idx) > 0 else None
    
    lead_time = None
    if warning_time is not None and runaway_time is not None:
        lead_time = runaway_time - warning_time
        
    return warning_time, runaway_time, lead_time
