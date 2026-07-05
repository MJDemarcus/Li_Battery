import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LI-MVP • Thermal Runaway Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #58a6ff !important; font-weight: 600; }
    .hero-lead {
        background: rgba(33,38,45,0.9);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 28px 36px;
        text-align: center;
        margin-bottom: 20px;
    }
    .lead-number {
        font-size: 5rem;
        font-weight: 800;
        color: #3fb950;
        line-height: 1;
        margin: 8px 0;
    }
    .lead-label {
        font-size: 1.1rem;
        color: #8b949e;
        margin-bottom: 4px;
    }
    .lead-sub {
        font-size: 0.9rem;
        color: #8b949e;
        margin-top: 8px;
    }
    .state-hero {
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        border-left: 6px solid;
        margin-bottom: 16px;
    }
    .state-0 { border-left-color: #3fb950; background: rgba(63,185,80,0.08); }
    .state-1 { border-left-color: #d29922; background: rgba(210,153,34,0.08); }
    .state-2 { border-left-color: #db6d28; background: rgba(219,109,40,0.10); }
    .state-3 { border-left-color: #f85149; background: rgba(248,81,73,0.12); }
    .state-4 { border-left-color: #ff7b72; background: rgba(248,81,73,0.22); }
    .state-name { font-size: 2rem; font-weight: 700; margin: 8px 0; }
    .state-prob { font-size: 1.2rem; color: #8b949e; }
    .trigger-row { display: flex; align-items: flex-start; gap: 14px; padding: 10px 0; border-bottom: 1px solid #21262d; }
    .trigger-dot { width: 14px; height: 14px; border-radius: 50%; margin-top: 3px; flex-shrink: 0; }
    .dot-active { background: #f85149; }
    .dot-inactive { background: #30363d; }
    .trigger-label { font-family: monospace; font-size: 0.85rem; font-weight: 700; color: #58a6ff; min-width: 80px; }
    .trigger-desc { font-size: 0.88rem; color: #8b949e; line-height: 1.45; }
    .trigger-desc-active { color: #c9d1d9; }
    .section-sep { border: none; border-top: 1px solid #21262d; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# ─── TRIGGER ENGINE (mirrored from app.py v2) ───────────────────────────────

def apply_noise_smoothing(df):
    df = df.copy()
    T_s = df["temperature"].rolling(window=10, min_periods=1).mean()
    df["temp_smooth"] = T_s
    dt = df["time"].diff().fillna(1.0).replace(0, 1.0)
    dT = T_s.diff().fillna(0.0)
    df["dT_dt"] = (dT / dt).rolling(window=10, min_periods=1).mean()
    d2T = df["dT_dt"].diff().fillna(0.0)
    df["d2T_dt2"] = (d2T / dt).rolling(window=10, min_periods=1).mean()
    return df

def evaluate_trigger_a(df):
    rm = df["temp_smooth"].expanding(min_periods=20).mean()
    rs = df["temp_smooth"].expanding(min_periods=20).std().fillna(0)
    return (df["temperature"] > rm + np.maximum(3 * rs, 1.5)).astype(int).cummax()

def evaluate_trigger_b(df, hthr=0.5, dthr=0.5, win=10):
    steep = (df["dT_dt"] > hthr).astype(int)
    return (steep.rolling(win, min_periods=1).mean() >= dthr).astype(int).cummax()

def evaluate_trigger_c(df, thr=100):
    sig = np.maximum(0, df["d2T_dt2"] * 100) ** 2
    return (sig > thr).astype(int).cummax(), sig

def evaluate_trigger_d(df, frac=0.25, sigma=2.5, T_amb=None):
    if T_amb is None:
        T_amb = df["temperature"].iloc[:50].min()
    n = max(50, int(len(df) * frac))
    early = df.iloc[:n]
    X = early["temperature"].values - T_amb
    rate = early["dT_dt"].fillna(0.0).values
    if len(X) > 1 and len(rate) > 1:
        try:
            valid_mask = np.isfinite(X) & np.isfinite(rate)
            if valid_mask.sum() > 1:
                coeffs = np.polyfit(X[valid_mask], rate[valid_mask], 1)
                beta, alpha = -coeffs[0], coeffs[1]
            else:
                alpha, beta = 0.0, 0.0
        except Exception:
            alpha, beta = 0.0, 0.0
    else:
        alpha, beta = 0.0, 0.0
    Xf = df["temperature"].values - T_amb
    dev = np.abs(df["dT_dt"].fillna(0.0).values - (alpha - beta * Xf))
    std = np.std(dev[:n]) + 1e-6
    anom = pd.Series(dev > sigma * std, index=df.index)
    anom.iloc[:n] = False
    return anom.astype(int).cummax(), alpha, beta, T_amb, dev

def get_state(a, b, c, d):
    total = int(a) + int(b) + int(c) + int(d)
    labels = {0: "STABLE", 1: "EXPLANATION NEEDED", 2: "WATCHING BRIEF", 3: "HIGH RISK", 4: "CRITICAL WARNING"}
    probs  = {0: 0, 1: 25, 2: 50, 3: 75, 4: 100}
    return total, probs[total], labels[total]

def simulate_prevention(df, t_warn, alpha, beta, T_amb):
    if t_warn is None:
        return None
    t_sim = np.arange(t_warn, t_warn + 300, 1.0)
    match = df[df["time"] >= t_warn]
    if match.empty:
        return None
    T_sim = np.zeros(len(t_sim))
    T_sim[0] = match["temperature"].iloc[0]
    sb = max(beta, 0.01)
    for i in range(1, len(t_sim)):
        T_sim[i] = T_sim[i - 1] + (0.0 - sb * 1.8 * (T_sim[i - 1] - T_amb))
    return pd.DataFrame({"time": t_sim, "temp_averted": T_sim})

# ─── TRIGGER EXPLANATIONS ────────────────────────────────────────────────────

TRIGGER_EXPLAIN = {
    "A": {
        "name": "Statistical Envelope",
        "inactive": "Temperature within historical bounds. Expanding ±3σ window shows normal behaviour.",
        "active":   "Temperature has exited the statistical envelope — reading above 3σ of the expanding baseline. Ambient conditions have changed or an anomaly has begun.",
    },
    "B": {
        "name": "Rate Density",
        "inactive": "Heating rate below threshold. dT/dt is not sustaining dangerous acceleration.",
        "active":   "Sustained heating rate above 0.5°C/s for ≥50% of the last 30 readings. The battery is in persistent acceleration — not a transient spike.",
    },
    "C": {
        "name": "2nd Derivative Spike",
        "inactive": "No thermal acceleration spike detected. Rate of change is stable.",
        "active":   "Second derivative of temperature has crossed the non-linear threshold — the rate of heating is itself accelerating. This catches the inflection point before sustained heat (Trigger B) becomes visible.",
    },
    "D": {
        "name": "Physics ODE Residual",
        "inactive": "Thermal behaviour matches Newton's cooling model. No excess heat generation detected.",
        "active":   "Actual dT/dt deviates >2.5σ from the physics model (dT/dt = α − β·ΔT). The battery is generating more heat than the lumped thermal model predicts — a signal of internal fault before temperature has risen enough for statistical triggers to fire.",
    },
}

# ─── DATA LOADING ────────────────────────────────────────────────────────────

DATASETS = {
    "Liguard Live Demo Dataset (BMS Logs)": "data/bms_simulations/Liguard_Live_Demo_Dataset.csv",
    "Synthetic Runaway (7-min lead-up)": "data/thermal_runaway_data.csv",
    "Battery Archive – Normal Cycling":  "data/external/battery_archive_cycling.csv",
    "NASA – Baseline Aging Sample":      "data/external/nasa_aging_sample.csv",
    "Sample Battery Data (Nominal)":     "data/sample_battery_data.csv",
    "NREL Abuse Test Sample":            "data/external/nrel_abuse_test_sample.csv",
}

COL_MAP = {
    "temp": "temperature", "t_c": "temperature", "t": "temperature",
    "timestamp": "time", "test_time (s)": "time",
    "temperature_measured": "temperature", "cycle": "time",
}

def load_dataset(path):
    base = os.path.dirname(__file__)
    for candidate in [path, os.path.join(base, path), os.path.join(base, "..", path)]:
        if os.path.exists(candidate):
            # If it's the live demo dataset, don't use comment="#" because of column headers starting with "#"
            comment_char = None if "Liguard_Live_Demo_Dataset" in candidate else "#"
            df = pd.read_csv(candidate, comment=comment_char)
            return df
    return None

def make_demo_df():
    t = np.arange(0, 600, 1.0)
    temp = 25 + 0.02 * t + 5 * np.exp((t - 450) / 40) * (t > 450) + np.random.normal(0, 0.3, len(t))
    return pd.DataFrame({"time": t, "temperature": temp})

def process_and_clean_df(df):
    # Time column parsing, sorting, and duplicate collapsing
    time_candidates = [c for c in df.columns if c.lower() in ['_time', 'time', 'timestamp', 'test_time (s)']]
    if time_candidates:
        time_col = time_candidates[0]
        if df[time_col].dtype == 'object' or isinstance(df[time_col].dtype, pd.DatetimeTZDtype) or df[time_col].dtype == 'datetime64[ns]':
            df['_time_parsed'] = pd.to_datetime(df[time_col], format='mixed', errors='coerce')
            df = df.dropna(subset=['_time_parsed'])
            df = df.sort_values('_time_parsed').reset_index(drop=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.difference(['_time_parsed'])
            agg_dict = {col: 'mean' for col in numeric_cols}
            for col in non_numeric_cols:
                agg_dict[col] = 'first'
            df = df.groupby('_time_parsed', as_index=False).agg(agg_dict)
            df['time'] = (df['_time_parsed'] - df['_time_parsed'].iloc[0]).dt.total_seconds()
        else:
            if time_col != 'time':
                df.rename(columns={time_col: 'time'}, inplace=True)
            df = df.sort_values('time').reset_index(drop=True)
    else:
        df.columns = df.columns.str.lower()
        for old, new in COL_MAP.items():
            if old in df.columns and new not in df.columns:
                df.rename(columns={old: new}, inplace=True)
        if 'time' not in df.columns:
            df['time'] = np.arange(len(df))

    # Temperature channel detection and selection
    temp_cols = [c for c in df.columns if 'temp' in c.lower() or c.lower() in ['t_c', 't', 'temperature_measured']]
    if temp_cols:
        if len(temp_cols) > 1:
            selected_temp_col = st.sidebar.selectbox("Select Temperature Channel", temp_cols, key="mc_temp_select_app")
        else:
            selected_temp_col = temp_cols[0]
    else:
        numeric = df.select_dtypes(include=[np.number]).columns
        selected_temp_col = numeric[-1] if len(numeric) > 0 else 'temperature'
        if selected_temp_col not in df.columns:
            df[selected_temp_col] = 0.0

    df['temperature'] = df[selected_temp_col]
    df['temperature'] = df['temperature'].replace(0.0, np.nan)
    df['temperature'] = df['temperature'].ffill().bfill()
    
    df = df[pd.to_numeric(df["time"], errors="coerce").notnull()].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["time", "temperature"])
    return df

# ─── SIDEBAR: dataset selection ──────────────────────────────────────────────

st.sidebar.title("🔋 LI-MVP v2")
st.sidebar.markdown("**Marsham Edge** | Physics State Machine")
st.sidebar.markdown("---")

data_source = st.sidebar.radio("Data source", ["Preloaded", "Upload CSV"])

if data_source == "Preloaded":
    selected = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
    df_raw = load_dataset(DATASETS[selected])
    if df_raw is None:
        st.sidebar.warning("Dataset not found — using demo trace.")
        df_raw = make_demo_df()
    df = process_and_clean_df(df_raw)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV (needs 'time' + 'temperature')", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df = process_and_clean_df(df_raw)
    else:
        st.sidebar.info("No file uploaded — using demo trace.")
        df = process_and_clean_df(make_demo_df())

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown("Four-trigger / five-state thermal runaway early-warning system. Trigger D (physics ODE residual) provides measurable lead time over simple threshold alarms.")

# ─── RUN ENGINE ─────────────────────────────────────────────────────────────

df = apply_noise_smoothing(df)
trig_a = evaluate_trigger_a(df)
trig_b = evaluate_trigger_b(df)
trig_c, spike_sig = evaluate_trigger_c(df)
trig_d, alpha, beta, T_amb, dev = evaluate_trigger_d(df)

results = [get_state(trig_a.iloc[i], trig_b.iloc[i], trig_c.iloc[i], trig_d.iloc[i]) for i in range(len(df))]
df["state"] = [r[0] for r in results]
df["prob"]  = [r[1] for r in results]

RUNAWAY_TEMP = 80.0
simple_alarm_rows = df[df["temperature"] >= RUNAWAY_TEMP]
system_alert_rows = df[df["state"] >= 2]  # Watching Brief or worse

t_simple  = float(simple_alarm_rows["time"].iloc[0])  if not simple_alarm_rows.empty  else None
t_system  = float(system_alert_rows["time"].iloc[0])  if not system_alert_rows.empty  else None
t_warn    = t_system

lead_secs = (t_simple - t_system) if (t_simple and t_system and t_system < t_simple) else None
averted_df = simulate_prevention(df, t_warn, alpha, beta, T_amb)

final_state = int(df["state"].iloc[-1])
final_prob  = int(df["prob"].iloc[-1])
state_label = {0: "STABLE", 1: "EXPLANATION NEEDED", 2: "WATCHING BRIEF", 3: "HIGH RISK", 4: "CRITICAL WARNING"}[final_state]
final_triggers = {
    "A": bool(trig_a.iloc[-1]),
    "B": bool(trig_b.iloc[-1]),
    "C": bool(trig_c.iloc[-1]),
    "D": bool(trig_d.iloc[-1]),
}

# ─── HERO: LEAD-TIME COMPARISON ─────────────────────────────────────────────

st.title("🔋 LI-MVP v2 – Physics State Machine")

TRIG_COLORS = {"A": "#58a6ff", "B": "#d29922", "C": "#db6d28", "D": "#3fb950"}

col_hero, col_state = st.columns([3, 2], gap="large")

with col_hero:
    # Build lead-time chart
    fig = go.Figure()

    # Temperature trace
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["temperature"],
        name="Temperature (°C)",
        line=dict(color="#ff7b72", width=2.5),
    ))

    # Averted path
    if averted_df is not None:
        fig.add_trace(go.Scatter(
            x=averted_df["time"], y=averted_df["temp_averted"],
            name="Averted path (intervention)",
            line=dict(color="#3fb950", width=3, dash="dash"),
        ))

    # System alert line
    if t_system:
        first_state = state_label if not system_alert_rows.empty else "Alert"
        fig.add_vline(
            x=t_system,
            line_width=2.5, line_dash="dash", line_color="#3fb950",
            annotation_text=f"System: {first_state if not system_alert_rows.empty else 'Alert'}",
            annotation_font=dict(color="#3fb950", size=12),
            annotation_position="top left",
        )

    # 80°C alarm line
    if t_simple:
        fig.add_vline(
            x=t_simple,
            line_width=2, line_dash="dot", line_color="#ff7b72",
            annotation_text="80°C threshold alarm",
            annotation_font=dict(color="#ff7b72", size=12),
            annotation_position="top right",
        )

    # Individual trigger lines (thin, annotated)
    for letter, series in [("A", trig_a), ("B", trig_b), ("C", trig_c), ("D", trig_d)]:
        first_pts = df[series == 1]
        if not first_pts.empty:
            t_first = float(first_pts["time"].iloc[0])
            fig.add_vline(
                x=t_first,
                line_width=1, line_dash="dot",
                line_color=TRIG_COLORS[letter],
                annotation_text=f"Trig {letter}",
                annotation_font=dict(color=TRIG_COLORS[letter], size=10),
                annotation_position="bottom left",
            )

    if lead_secs:
        title_text = f"Lead-time advantage: <b>{lead_secs:.0f}s earlier</b> than simple threshold alarm"
    else:
        title_text = "Thermal runaway detection trace"

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title_text, font=dict(size=15, color="#c9d1d9")),
        xaxis_title="Time (s)",
        yaxis_title="Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=60, b=40),
        height=420,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Lead-time callout
    if lead_secs:
        st.markdown(f"""
        <div class="hero-lead">
            <div class="lead-label">System alert fired before threshold alarm</div>
            <div class="lead-number">{lead_secs:.0f}s</div>
            <div class="lead-label">earlier</div>
            <div class="lead-sub">
                System triggered <b>Watching Brief</b> at t={t_system:.0f}s &nbsp;·&nbsp;
                Simple 80°C alarm at t={t_simple:.0f}s
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif t_system and not t_simple:
        st.markdown(f"""
        <div class="hero-lead">
            <div class="lead-label">System flagged anomaly — temperature has not yet reached 80°C threshold</div>
            <div class="lead-number">Pre-emptive</div>
            <div class="lead-sub">System alert at t={t_system:.0f}s &nbsp;·&nbsp; 80°C threshold: not reached</div>
        </div>
        """, unsafe_allow_html=True)

# ─── STATE PANEL ─────────────────────────────────────────────────────────────

with col_state:
    st.markdown(f"""
    <div class="state-hero state-{final_state}">
        <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase; letter-spacing:1px;">Current State</div>
        <div class="state-name">{state_label}</div>
        <div class="state-prob">TR probability: <b>{final_prob}%</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Active triggers**")
    for letter in ["A", "B", "C", "D"]:
        active = final_triggers[letter]
        dot_cls = "dot-active" if active else "dot-inactive"
        desc_cls = "trigger-desc-active" if active else "trigger-desc"
        desc = TRIGGER_EXPLAIN[letter]["active" if active else "inactive"]
        status_text = "● ACTIVE" if active else "○ clear"
        status_color = "#f85149" if active else "#484f58"
        st.markdown(f"""
        <div class="trigger-row">
            <div>
                <div class="trigger-label" style="color:{TRIG_COLORS[letter]}">
                    {letter} — {TRIGGER_EXPLAIN[letter]['name']}
                    <span style="color:{status_color}; font-size:0.75rem; margin-left:8px;">{status_text}</span>
                </div>
                <div class="{desc_cls}">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── DETAIL SECTION ─────────────────────────────────────────────────────────

st.markdown('<hr class="section-sep">', unsafe_allow_html=True)

with st.expander("Prevention simulation & raw metrics", expanded=False):
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("State", state_label)
    with m2:
        st.metric("TR Probability", f"{final_prob}%")
    with m3:
        st.metric("System alert at", f"{t_system:.0f}s" if t_system else "—")
    with m4:
        st.metric("Lead time", f"{lead_secs:.0f}s" if lead_secs else "—")

    if averted_df is not None:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["time"], y=df["temperature"], name="Actual", line=dict(color="#ff7b72", width=2)))
        fig2.add_trace(go.Scatter(x=averted_df["time"], y=averted_df["temp_averted"], name="Averted (intervention at Watching Brief)", line=dict(color="#3fb950", width=3, dash="dash")))
        fig2.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=30, b=30), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
        st.plotly_chart(fig2, use_container_width=True)
        st.info("Averted path simulates immediate power-derating + active cooling triggered at 'Watching Brief' state.")

# Secondary visualization: Electrical Load Profile (if columns are present)
voltage_col = next((c for c in df.columns if 'average_line_to_line_volts' in c.lower() or 'line_to_line_volts' in c.lower()), None)
current_col = next((c for c in df.columns if 'average_line_current' in c.lower() or 'line_current' in c.lower()), None)

if voltage_col or current_col:
    st.markdown('<hr class="section-sep">', unsafe_allow_html=True)
    st.markdown("### ⚡ Synchronized Electrical Load Profile")
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
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
    )
    st.plotly_chart(fig_elec, use_container_width=True)
