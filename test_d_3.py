import pandas as pd
import numpy as np
def apply_smoothing(df, window=10):
    df_smooth = df.copy()
    df_smooth['temp_smooth'] = df['temperature'].rolling(window=window, min_periods=1).mean()
    return df_smooth

df_raw = pd.read_csv('data/thermal_runaway_data.csv', comment='#')
df_raw.columns = df_raw.columns.str.lower()
col_map = {'temp': 'temperature', 't_c': 'temperature', 't': 'temperature', 'timestamp': 'time', 'test_time (s)': 'time'}
for old_col, new_col in col_map.items():
    if old_col in df_raw.columns and new_col not in df_raw.columns:
        df_raw.rename(columns={old_col: new_col}, inplace=True)
if 'time' not in df_raw.columns: df_raw['time'] = np.arange(len(df_raw))

df = apply_smoothing(df_raw)

def evaluate_trigger_d(df, early_fraction=0.25, dev_sigma=2.5):
    T_amb = df['temperature'].iloc[:50].min()
    n_early = max(50, int(len(df) * early_fraction))
    early = df.iloc[:n_early]
    
    dt = early['time'].diff().fillna(1.0)
    dT = early['temp_smooth'].diff().fillna(0.0)
    rate = dT / dt
    X = early['temperature'].values - T_amb
    
    coeffs = np.polyfit(X, rate.values, 1)
    beta, alpha = -coeffs[0], coeffs[1]
    
    X_full = df['temperature'].values - T_amb
    pred_rate = alpha - beta * X_full
    actual_rate = (df['temp_smooth'].diff().fillna(0.0)) / (df['time'].diff().fillna(1.0))
    deviation = np.abs(actual_rate - pred_rate)
    
    std_res = np.std(deviation[:n_early]) + 1e-6
    raw_trigger = deviation > dev_sigma * std_res
    raw_trigger[:n_early] = False
    trigger = pd.Series(raw_trigger.astype(int))
    return trigger, alpha, beta, T_amb

trig_d, a,b, t_amb = evaluate_trigger_d(df)
t_first = df[trig_d == 1]['time'].iloc[0] if len(df[trig_d==1]) > 0 else None
print("t_first =", t_first)
print("n_early =", max(50, int(len(df) * 0.25)))
print("trig_d first 10 vals: ", trig_d.head(10).tolist())
