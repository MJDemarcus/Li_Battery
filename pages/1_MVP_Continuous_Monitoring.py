import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ----------------- UI CONFIG & THEME -----------------
st.set_page_config(page_title="Marsham Edge | Li-Battery TR Predictor", page_icon="🔋", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #58a6ff !important; }
    .state-box { background: rgba(33, 38, 45, 0.85); border-radius: 12px; padding: 20px; text-align: center; border-left: 5px solid; margin: 10px 0; }
    .state-0 { border-left-color: #3fb950; } .state-1 { border-left-color: #d29922; }
    .state-2 { border-left-color: #f85149; } .state-3 { border-left-color: #db6d28; }
    .state-4 { border-left-color: #ff7b72; background: rgba(248,81,73,0.15); }
    .trigger-badge { display: inline-block; background: #21262d; border-radius: 12px; padding: 4px 10px; margin: 2px; font-size: 0.75rem; }
    .trigger-active { background: #f85149; color: white; }
    .metric-value { font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ----------------- CORE ENGINE: TRIGGERS A-D -----------------

def apply_smoothing(df, window=10):
    df_smooth = df.copy()
    df_smooth['temp_smooth'] = df['temperature'].rolling(window=window, min_periods=1).mean()
    return df_smooth

def evaluate_trigger_a(df):
    """Trigger A: Causal Statistical Envelope (Replaces Prophet for Edge)"""
    # Moving average and std dev to avoid future leakage
    rolling_mean = df['temperature'].expanding(min_periods=20).mean()
    rolling_std = df['temperature'].expanding(min_periods=20).std()
    upper_bound = rolling_mean + (2 * rolling_std)
    trigger = (df['temperature'] > upper_bound).astype(int)
    return trigger

def evaluate_trigger_b(df, heating_threshold=0.5, density_threshold=0.5, window=10):
    """Trigger B: Heating Density"""
    dt = df['time'].diff().fillna(1.0)
    # Prevent division by zero
    dt[dt == 0] = 1.0
    dT = df['temp_smooth'].diff().fillna(0.0)
    rate = dT / dt
    steep = (rate > heating_threshold).astype(int)
    density = steep.rolling(window=window, min_periods=1).mean()
    return (density >= density_threshold).astype(int)

def evaluate_trigger_c(df, threshold=100):
    """Trigger C: Acceleration Spike (2nd Derivative)"""
    dt = df['time'].diff().fillna(1.0)
    # Prevent division by zero
    dt[dt == 0] = 1.0
    dT = df['temp_smooth'].diff().fillna(0.0)
    d2T = dT.diff().fillna(0.0) / dt
    spike_signal = np.maximum(0, d2T * 100) ** 2
    return (spike_signal > threshold).astype(int), spike_signal

def evaluate_trigger_d(df, early_fraction=0.25, dev_sigma=3.0):
    """Trigger D: Physics-Informed Digital Twin Residual"""
    T_amb = df['temperature'].iloc[:50].min()
    n_early = max(50, int(len(df) * early_fraction))
    early = df.iloc[:n_early]
    
    dt = early['time'].diff().fillna(1.0)
    # Prevent division by zero
    dt[dt == 0] = 1.0
    dT = early['temp_smooth'].diff().fillna(0.0)
    rate = dT / dt
    X = early['temperature'].values - T_amb
    
    try:
        valid_mask = np.isfinite(X) & np.isfinite(rate.values)
        if valid_mask.sum() > 1:
            coeffs = np.polyfit(X[valid_mask], rate.values[valid_mask], 1) # rate = α - βX
            beta, alpha = -coeffs[0], coeffs[1]
        else:
            alpha, beta = 0.0, 0.0
    except Exception:
        alpha, beta = 0.0, 0.0
    
    # Full trace evaluation
    X_full = df['temperature'].values - T_amb
    pred_rate = alpha - beta * X_full
    actual_rate = (df['temp_smooth'].diff().fillna(0.0)) / (df['time'].diff().fillna(1.0))
    deviation = np.abs(actual_rate - pred_rate)
    
    std_res = np.std(deviation[:n_early]) + 1e-6
    raw_trigger = deviation > dev_sigma * std_res
    raw_trigger[:n_early] = False
    trigger = pd.Series(raw_trigger.astype(int))
    return trigger, alpha, beta, T_amb

def get_state(a, b, c, d):
    total = a + b + c + d
    labels = {0: "STABLE", 1: "INVESTIGATE", 2: "WATCHING BRIEF", 3: "HIGH RISK", 4: "CRITICAL"}
    probs = {0: 0, 1: 25, 2: 50, 3: 75, 4: 100}
    return total, probs[total], labels[total]

def simulate_prevention(df, t_warning, alpha, beta, T_amb):
    """Averted Trajectory Simulation"""
    if t_warning is None: return None
    t_sim = np.arange(t_warning, t_warning + 300, 1.0)
    if len(df[df['time'] >= t_warning]) == 0:
        return None
        
    T_sim = np.zeros(len(t_sim))
    T_sim[0] = df[df['time'] >= t_warning]['temperature'].iloc[0]
    
    # Intervention: Stop heat gen (α=0) + Active Cooling (β * 1.8). Force beta positive.
    safe_beta = max(beta, 0.01)
    
    for i in range(1, len(t_sim)):
        dTdt = 0.0 - (safe_beta * 1.8) * (T_sim[i-1] - T_amb)
        T_sim[i] = T_sim[i-1] + dTdt
    return pd.DataFrame({'time': t_sim, 'temp_averted': T_sim})

# ----------------- MAIN APP LOGIC -----------------

st.sidebar.title("Marsham Edge | BMS v2")
st.title("🔋 Lithium Battery Thermal Runaway Predictor")

st.markdown("### 📥 Dataset Ingestion")
data_source = st.radio("Choose Data Source", ["Preloaded Datasets", "Upload CSV Dataset"])

if data_source == "Preloaded Datasets":
    datasets = {
        "Liguard Live Demo Dataset (BMS Logs)": "data/bms_simulations/Liguard_Live_Demo_Dataset.csv",
        "Synthetic Baseline (7m Lead-Up)": "data/thermal_runaway_data.csv",
        "Battery Archive - Normal Cycling": "data/external/battery_archive_cycling.csv",
        "NASA - Baseline Aging Sample": "data/external/nasa_aging_sample.csv",
        "Sample Battery Data (Nominal)": "data/sample_battery_data.csv",
        "NREL Abuse Test Sample": "data/external/nrel_abuse_test_sample.csv"
    }
    selected_ds = st.selectbox("Select a Dataset:", list(datasets.keys()))
    
    # Simple relative pathing depending on where streamlit is run
    path = datasets[selected_ds]
    import os
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), '..', path)
        
    try:
        comment_char = None if "Liguard_Live_Demo_Dataset" in path else "#"
        df_raw = pd.read_csv(path, comment=comment_char)
    except Exception as e:
        st.error(f"Failed to load preloaded set: {e}")
        st.stop()
        
else:
    uploaded_file = st.file_uploader("Upload a BMS CSV dataset (must contain 'time' and 'temperature')", type=['csv'])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
    else:
        st.info("Waiting for file upload... Using Demo Engine in background.")
        t = np.arange(0, 600, 1)
        temp = 25 + 0.02*t + 5 * np.exp((t-450)/40) * (t > 450) + np.random.normal(0, 0.3, len(t))
        df_raw = pd.DataFrame({'time': t, 'temperature': temp})

# Time column parsing, sorting, and duplicate collapsing
time_candidates = [c for c in df_raw.columns if c.lower() in ['_time', 'time', 'timestamp', 'test_time (s)']]
if time_candidates:
    time_col = time_candidates[0]
    df_raw['_time_parsed'] = pd.to_datetime(df_raw[time_col], format='mixed', errors='coerce')
    df_raw = df_raw.dropna(subset=['_time_parsed'])
    df_raw = df_raw.sort_values('_time_parsed').reset_index(drop=True)
    
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df_raw.select_dtypes(exclude=[np.number]).columns.difference(['_time_parsed'])
    agg_dict = {col: 'mean' for col in numeric_cols}
    for col in non_numeric_cols:
        agg_dict[col] = 'first'
    df_raw = df_raw.groupby('_time_parsed', as_index=False).agg(agg_dict)
    df_raw['time'] = (df_raw['_time_parsed'] - df_raw['_time_parsed'].iloc[0]).dt.total_seconds()
else:
    df_raw.columns = df_raw.columns.str.lower()
    col_map = {'temp': 'temperature', 't_c': 'temperature', 'timestamp': 'time', 'test_time (s)': 'time'}
    for old, new in col_map.items():
        if old in df_raw.columns and new not in df_raw.columns:
            df_raw.rename(columns={old: new}, inplace=True)
    if 'time' not in df_raw.columns:
        df_raw['time'] = np.arange(len(df_raw))

# Temperature channel detection and selection
temp_cols = [c for c in df_raw.columns if 'temp' in c.lower() or c.lower() in ['t_c', 't', 'temperature_measured']]
if temp_cols:
    if len(temp_cols) > 1:
        selected_temp_col = st.sidebar.selectbox("Select Temperature Channel", temp_cols, key="mc_temp_select")
    else:
        selected_temp_col = temp_cols[0]
else:
    numeric = df_raw.select_dtypes(include=[np.number]).columns
    selected_temp_col = numeric[-1] if len(numeric) > 0 else 'temperature'
    if selected_temp_col not in df_raw.columns:
        df_raw[selected_temp_col] = 0.0

# Set 'temperature' column to the cleaned chosen column
df_raw['temperature'] = df_raw[selected_temp_col]

# Clean chosen temperature: replace 0.0 and NaN with NaN, then ffill/bfill to remove dropouts
df_raw['temperature'] = df_raw['temperature'].replace(0.0, np.nan)
df_raw['temperature'] = df_raw['temperature'].ffill().bfill()

df = apply_smoothing(df_raw)

# Run Triggers & Force Stickiness (Once TR starts, it doesn't un-start)
trig_a = evaluate_trigger_a(df).cummax()
trig_b = evaluate_trigger_b(df).cummax()
trig_c, spike = evaluate_trigger_c(df)
trig_c = trig_c.cummax()
trig_d, alpha, beta, T_amb = evaluate_trigger_d(df, dev_sigma=2.5) 
trig_d = trig_d.cummax()

# Build State Data
results = [get_state(trig_a.iloc[i], trig_b.iloc[i], trig_c.iloc[i], trig_d.iloc[i]) for i in range(len(df))]
df['state'], df['prob'] = [r[0] for r in results], [r[1] for r in results]

# Prevention
t_warn = df[df['state'] >= 2]['time'].iloc[0] if any(df['state'] >= 2) else None
averted_df = simulate_prevention(df, t_warn, alpha, beta, T_amb)

# Metrics
final_state = df["state"].iloc[-1]
final_prob = df["prob"].iloc[-1]
health_labels = {0: "Healthy (Nominal)", 1: "Degraded (Investigate)", 2: "Unstable (Action Required)", 3: "Critical Risk", 4: "Terminal (Runaway)"}

st.markdown("### ⚕️ Health Diagnostic & Prognosis")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="state-box state-{final_state}"><small>DIAGNOSTIC</small><div style="font-size: 1.4rem; font-weight: 700; margin-top: 10px;">{health_labels[final_state]}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="state-box state-{final_state}"><small>TR PROBABILITY</small><div class="metric-value">{final_prob}%</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="state-box"><small>INTERVENTION WINDOW</small><div class="metric-value">{t_warn if t_warn else 0:.0f}s</div><small>Lead Time</small></div>', unsafe_allow_html=True)
with c4:
    active = [t for t, v in zip("ABCD", [trig_a.iloc[-1], trig_b.iloc[-1], trig_c.iloc[-1], trig_d.iloc[-1]]) if v]
    badges = "".join([f'<span class="trigger-badge trigger-active">Criteria {t}</span>' for t in active]) or "None Active"
    st.markdown(f'<div class="state-box"><small>TRIGGER CODES</small><div>{badges}</div></div>', unsafe_allow_html=True)

# Visualization
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['time'], y=df['temperature'], name="Actual Temp", line=dict(color="#ff7b72", width=3)))

colors = {"A": "#58a6ff", "B": "#d29922", "C": "#db6d28", "D": "#3fb950"}
trigger_series = {"A": trig_a, "B": trig_b, "C": trig_c, "D": trig_d}

for trig_letter, series in trigger_series.items():
    trig_points = df[series == 1]
    if not trig_points.empty:
        t_first = trig_points.iloc[0]['time']
        fig.add_vline(x=t_first, line_width=1.5, line_dash="dash", line_color=colors[trig_letter], annotation_text=f"Trig {trig_letter}", annotation_position="top left")

if averted_df is not None:
    fig.add_trace(go.Scatter(x=averted_df['time'], y=averted_df['temp_averted'], name="Averted Path (Intervention)", line=dict(color="#3fb950", width=4, dash='dash')))

fig.update_layout(template="plotly_dark", title="Detection vs. Prevention Trajectory", xaxis_title="Time (s)", yaxis_title="Temp (°C)")
st.plotly_chart(fig, use_container_width=True)

# Secondary visualization: Electrical Load Profile (if columns are present)
voltage_col = next((c for c in df.columns if 'average_line_to_line_volts' in c.lower() or 'line_to_line_volts' in c.lower()), None)
current_col = next((c for c in df.columns if 'average_line_current' in c.lower() or 'line_current' in c.lower()), None)

if voltage_col or current_col:
    st.markdown("### Synchronized Electrical Load Profile")
    fig_elec = go.Figure()
    
    if voltage_col:
        fig_elec.add_trace(go.Scatter(x=df['time'], y=df[voltage_col], name="Voltage (V)", line=dict(color="#58a6ff", width=2)))
    if current_col:
        fig_elec.add_trace(go.Scatter(x=df['time'], y=df[current_col], name="Current (A)", line=dict(color="#d29922", width=2), yaxis="y2"))
        
    fig_elec.update_layout(
        title="BMS Electrical Monitoring",
        template="plotly_dark",
        xaxis_title="Time (seconds)",
        yaxis=dict(title=dict(text="Voltage (V)", font=dict(color="#58a6ff")), tickfont=dict(color="#58a6ff")),
        yaxis2=dict(title=dict(text="Current (A)", font=dict(color="#d29922")), tickfont=dict(color="#d29922"), overlaying="y", side="right"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_elec, use_container_width=True)

st.info("**Groundbreaking Feature:** The green dashed line simulates immediate power-derating and active cooling at the 'Watching Brief' stage.")
