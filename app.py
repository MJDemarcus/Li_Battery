import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ----------------- UI CONFIG -----------------
st.set_page_config(page_title="LI-MVP • Thermal Runaway Predictor", page_icon="🔋", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #58a6ff !important; font-weight: 600; }
    .state-box {
        background: rgba(33, 38, 45, 0.85);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border-left: 5px solid;
        margin: 10px 0;
    }
    .state-0 { border-left-color: #3fb950; }
    .state-1 { border-left-color: #d29922; }
    .state-2 { border-left-color: #f85149; }
    .state-3 { border-left-color: #ff7b72; background: rgba(248,81,73,0.15); }
    .state-4 { border-left-color: #ff7b72; background: rgba(248,81,73,0.25); }
    .trigger-badge { display: inline-block; background: #21262d; border-radius: 20px; padding: 4px 12px; margin: 4px; font-size: 0.85rem; font-family: monospace; }
    .trigger-active { background: #f85149; color: white; }
    .metric-value { font-size: 2.4rem; font-weight: 700; margin: 8px 0; }
    .etr-box { border-left-color: #db6d28; background: rgba(219, 109, 40, 0.15); }
    .etr-text { font-size: 1.8rem; font-weight: 700; color: #db6d28; margin: 4px 0;}
</style>
""", unsafe_allow_html=True)

st.title("🔋 LI-MVP v2 – Discrete Physics State Machine")
st.caption("Early Thermal Runaway Prediction + Prevention Simulator | A/B/C/D Triggers")

import os


# ----------------- CORE FUNCTIONS (v1 + v2) -----------------
def apply_noise_smoothing(df):
    df = df.copy()
    T_smooth = df['temperature'].rolling(window=10, min_periods=1).mean()
    df['temp_smooth'] = T_smooth
    
    dt = df['time'].diff().fillna(1.0)
    dt[dt == 0] = 1.0
    
    dT_dt = T_smooth.diff().fillna(0.0) / dt
    df['dT_dt'] = dT_dt.rolling(window=10, min_periods=1).mean()
    
    d2T_dt2 = df['dT_dt'].diff().fillna(0.0) / dt
    df['d2T_dt2'] = d2T_dt2.rolling(window=10, min_periods=1).mean()
    return df

def evaluate_trigger_a(df):
    """Trigger A: Causal Statistical Envelope"""
    if 'temp_smooth' not in df.columns:
        df = apply_noise_smoothing(df)
    rolling_mean = df['temp_smooth'].expanding(min_periods=20).mean()
    rolling_std = df['temp_smooth'].expanding(min_periods=20).std().fillna(0)
    # Add a minimum noise floor (1.5C) so perfectly flat holds don't trigger on 0.01C sensor jitter
    allowed_upper = rolling_mean + np.maximum(3 * rolling_std, 1.5)
    trigger = (df['temperature'] > allowed_upper).astype(int).cummax()
    return pd.Series(trigger, index=df.index)

def evaluate_trigger_b(df, heating_threshold=0.5, density_threshold=0.5, window=10):
    if 'dT_dt' not in df.columns:
        df = apply_noise_smoothing(df)
    steep_heating = (df['dT_dt'] > heating_threshold).astype(int)
    density = steep_heating.rolling(window=window, min_periods=1).mean()
    trigger = (density >= density_threshold).astype(int).cummax()
    return pd.Series(trigger, index=df.index)

def evaluate_trigger_c(df, geometric_threshold=100):
    if 'd2T_dt2' not in df.columns:
        df = apply_noise_smoothing(df)
    spike_signal = np.maximum(0, df['d2T_dt2'] * 100) ** 2
    trigger = (spike_signal > geometric_threshold).astype(int).cummax()
    return pd.Series(trigger, index=df.index), spike_signal

def evaluate_trigger_d(df, early_fraction=0.25, dev_sigma=2.5, T_amb=None):
    """Physics-informed Trigger D – lumped thermal model residual"""
    if 'dT_dt' not in df.columns:
        df = apply_noise_smoothing(df)
    if T_amb is None:
        T_amb = df['temperature'].iloc[:50].min()
    n_early = max(50, int(len(df) * early_fraction))
    early = df.iloc[:n_early].copy()
    
    X = early['temperature'].values - T_amb
    actual_rate = early['dT_dt'].fillna(0.0).values
    
    if len(X) > 1 and len(actual_rate) > 1:
        coeffs = np.polyfit(X, actual_rate, 1)
        beta = -coeffs[0]
        alpha = coeffs[1]
    else:
        alpha, beta = 0.0, 0.0
        
    pred_rate_early = alpha - beta * X
    std_res = np.std(np.abs(actual_rate - pred_rate_early)) + 1e-6
    
    # Full dataset
    X_full = df['temperature'].values - T_amb
    pred_rate_full = alpha - beta * X_full
    actual_rate_full = df['dT_dt'].fillna(0.0).values
    deviation = np.abs(actual_rate_full - pred_rate_full)
    
    # Prevent early triggering
    is_anomaly = deviation > dev_sigma * std_res
    is_anomaly = pd.Series(is_anomaly, index=df.index)
    is_anomaly.iloc[:n_early] = False
    
    trigger = is_anomaly.astype(int).cummax()
    return pd.Series(trigger, index=df.index), alpha, beta, T_amb, deviation

def get_state_and_probability(a, b, c, d):
    total = a + b + c + d
    prob_map = {0: 0, 1: 25, 2: 50, 3: 75, 4: 100}
    probability = prob_map.get(total, 100)
    labels = {0: "STABLE RUN", 1: "EXPLANATION NEEDED", 2: "WATCHING BRIEF", 3: "HIGH RISK", 4: "CRITICAL WARNING"}
    return total, probability, labels[total]

def simulate_prevention(df, t_warning, alpha, beta, T_amb, sim_seconds=300):
    if t_warning is None:
        return None
    dt_sim = 1.0
    t_sim = np.arange(t_warning, t_warning + sim_seconds, dt_sim)
    T_sim = np.zeros(len(t_sim))
    
    # Safe index search
    matches = df[df['time'] >= t_warning]
    if len(matches) == 0:
        return None
    T_sim[0] = matches['temperature'].iloc[0]
    
    alpha_int = 0.0
    # Safe beta to prevent runaway explosion on Averted Path
    safe_beta = max(beta, 0.01)
    beta_int = safe_beta * 1.8
    for i in range(1, len(t_sim)):
        dTdt = alpha_int - beta_int * (T_sim[i-1] - T_amb)
        T_sim[i] = T_sim[i-1] + dTdt * dt_sim
    return pd.DataFrame({'time': t_sim, 'temp_averted': T_sim})

# ----------------- LOAD DATA -----------------
def load_data(uploaded_file=None, selected_sim=None):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    elif selected_sim:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bms_simulations", selected_sim)
        df = pd.read_csv(file_path)
    else:
        # fallback demo
        t = np.arange(0, 600, 1)
        temp = 25 + 0.02 * t
        df = pd.DataFrame({'time': t, 'temperature': temp})
        
    df.columns = df.columns.str.lower()
    col_map = {'temp': 'temperature', 't_c': 'temperature', 'timestamp': 'time', 'test_time (s)': 'time'}
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
            
    if 'time' not in df.columns:
        df['time'] = np.arange(len(df))
    if 'temperature' not in df.columns:
        numeric = df.select_dtypes(include=[np.number]).columns
        df['temperature'] = df[numeric[-1]]
        
    start_date = datetime(2024, 1, 1)
    df['ds'] = [start_date + timedelta(seconds=t) for t in df['time']]
    return df

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3665/3665977.png", width=70)
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload your BMS data (CSV/XLSX)", type=["csv", "xlsx"])

sim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bms_simulations")
sim_files = sorted([f for f in os.listdir(sim_dir) if f.endswith('.csv')]) if os.path.exists(sim_dir) else []

selected_sim = None
if not uploaded_file and sim_files:
    selected_sim = st.sidebar.selectbox("Select Pre-loaded BMS Dataset", sim_files)

# ----------------- MAIN APP -----------------
df = load_data(uploaded_file, selected_sim)

with st.spinner("Running A/B/C/D Triggers + Prevention Simulator..."):
    df = apply_noise_smoothing(df)

    trigger_a = evaluate_trigger_a(df)
    trigger_b = evaluate_trigger_b(df)
    trigger_c, spike_signal = evaluate_trigger_c(df)
    trigger_d, alpha_fit, beta_fit, T_amb_fit, deviation = evaluate_trigger_d(df)

    states = []
    probabilities = []
    for i in range(len(df)):
        state, prob, _ = get_state_and_probability(
            trigger_a.iloc[i], trigger_b.iloc[i], trigger_c.iloc[i], trigger_d.iloc[i]
        )
        states.append(state)
        probabilities.append(prob)

    df['trigger_a'] = trigger_a.values
    df['trigger_b'] = trigger_b.values
    df['trigger_c'] = trigger_c.values
    df['trigger_d'] = trigger_d.values
    df['state'] = states
    df['probability'] = probabilities
    df['spike_signal'] = spike_signal.values if hasattr(spike_signal, 'values') else spike_signal
    df['deviation'] = deviation if isinstance(deviation, np.ndarray) else deviation.values

    current_state = df['state'].iloc[-1]
    current_prob = df['probability'].iloc[-1]
    
    watching_mask = df['state'] >= 2
    t_warning = df.loc[watching_mask, 'time'].iloc[0] if watching_mask.any() else None
    t_critical = df[df['state'] == 4]['time'].iloc[0] if len(df[df['state'] == 4]) > 0 else None
    
    etr_text = None
    if current_prob >= 50:
        T_curr = df['temperature'].iloc[-1]
        v = df['dT_dt'].iloc[-1]
        a = df['d2T_dt2'].iloc[-1]
        T_crit = 300.0
        
        if T_curr >= T_crit:
            etr_text = "Already reached"
        else:
            if a > 0.001:
                desc = v**2 - 4 * (0.5 * a) * (T_curr - T_crit)
                if desc >= 0:
                    t1 = (-v + np.sqrt(desc)) / a
                    etr_text = f"~{max(0, t1 * 0.8):.0f} to {max(0, t1 * 1.2):.0f} sec"
            elif v > 0.01:
                t1 = (T_crit - T_curr) / v
                etr_text = f"~{max(0, t1 * 0.9):.0f} to {max(0, t1 * 1.1):.0f} sec"

    averted_df = simulate_prevention(df, t_warning, alpha_fit, beta_fit, T_amb_fit) if current_state >= 3 else None

# ----------------- DASHBOARD UI -----------------
cols = st.columns(5)
with cols[0]:
    st.markdown(f"""
    <div class="state-box state-{current_state}">
        <div>CURRENT STATE</div>
        <div class="metric-value">{current_state}/4</div>
        <div>{current_prob}% Probability of Runaway</div>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    active = [t for t, v in [("A", df['trigger_a'].iloc[-1]), ("B", df['trigger_b'].iloc[-1]),
                             ("C", df['trigger_c'].iloc[-1]), ("D", df['trigger_d'].iloc[-1])] if v]
    st.markdown(f"""
    <div class="state-box">
        <div>ACTIVE TRIGGERS</div>
        <div>{' '.join([f'<span class="trigger-badge trigger-active">{t}</span>' for t in active]) or "<span class='trigger-badge'>None</span>"}</div>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    st.markdown(f"""
    <div class="state-box">
        <div>FIRST [2/4] WARNING</div>
        <div class="metric-value">t = {t_warning:.0f}s</div>
        <div style="font-size:0.8rem;">← Lead time starts here</div>
    </div>
    """ if t_warning else '<div class="state-box"><div>FIRST WARNING</div><div>No warning yet</div></div>', unsafe_allow_html=True)

with cols[3]:
    st.markdown(f"""
    <div class="state-box">
        <div>CRITICAL [4/4] AT</div>
        <div class="metric-value">t = {t_critical:.0f}s</div>
    </div>
    """ if t_critical else '<div class="state-box"><div>CRITICAL</div><div>Not reached</div></div>', unsafe_allow_html=True)

with cols[4]:
    st.markdown(f"""
    <div class="state-box etr-box">
        <div>EST. TIME TO RUNAWAY (ETR)</div>
        <div class="etr-text">{etr_text}</div>
        <div style="font-size:0.8rem; color:#db6d28;">Kinematic Projection (300°C)</div>
    </div>
    """ if etr_text else '<div class="state-box"><div>EST. TIME TO RUNAWAY</div><div class="metric-value" style="color:#6e7681;">Safe</div><div style="font-size:0.8rem;">No imminent threat</div></div>', unsafe_allow_html=True)

# ----------------- VISUALIZATION -----------------
st.markdown("### Temperature Trace + A/B/C/D Triggers + Averted Trajectory")
fig = go.Figure()

fig.add_trace(go.Scatter(x=df['time'], y=df['temperature'], name="Measured Temperature (°C)", line=dict(color="#ff7b72", width=3)))

if averted_df is not None:
    fig.add_trace(go.Scatter(x=averted_df['time'], y=averted_df['temp_averted'], name="🚀 WITH INTERVENTION (averted)", line=dict(color="#3fb950", width=4, dash="dash")))

# State background
state_colors = {0: '#3fb950', 1: '#d29922', 2: '#f85149', 3: '#ff7b72', 4: '#ff0000'}
start = 0
for i in range(1, len(df)):
    if df['state'].iloc[i] != df['state'].iloc[i-1]:
        fig.add_vrect(x0=df['time'].iloc[start], x1=df['time'].iloc[i-1], fillcolor=state_colors[df['state'].iloc[start]], opacity=0.15, layer="below", line_width=0)
        start = i
fig.add_vrect(x0=df['time'].iloc[start], x1=df['time'].iloc[-1], fillcolor=state_colors[df['state'].iloc[start]], opacity=0.15, layer="below", line_width=0)

# Clean, single vertical line markers for A/B/C/D
colors = {"a": "#58a6ff", "b": "#d29922", "c": "#db6d28", "d": "#a371f7"}
for trig_letter in ["a", "b", "c", "d"]:
    trig_col = f"trigger_{trig_letter}"
    trig_points = df[df[trig_col] == 1]
    if not trig_points.empty:
        # Find the very first moment it triggered
        t_first = trig_points.iloc[0]['time']
        label = f"<b>Trigger {trig_letter.upper()}</b>"
        fig.add_vline(x=t_first, line_width=2, line_dash="dash", line_color=colors[trig_letter], 
                      annotation_text=label, annotation_position="top left", annotation_font=dict(color=colors[trig_letter], size=12))

fig.update_layout(title="State Machine + Prevention Simulator", template="plotly_dark", xaxis_title="Time (seconds)", yaxis_title="Temperature (°C)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
st.plotly_chart(fig, use_container_width=True)

st.success("✅ MVP ready! Upload your own BMS data or use the synthetic dataset to test forced thermal runaway.")
st.caption("Lead time measured from first [2/4] Watching Brief • Prevention simulation activates automatically at warning")
