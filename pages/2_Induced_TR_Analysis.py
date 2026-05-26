import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.hybrid_evaluator import run_hybrid_heuristic

st.set_page_config(page_title="Induced TR Analysis", page_icon="🔥", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stApp h1, .stApp h2, .stApp h3 { color: #58a6ff !important; }
    .metric-box { background: rgba(33,38,45,0.85); border:1px solid #30363d; border-radius:8px; padding:15px; text-align:center; }
    .lead-badge { font-size: 1.6rem; font-weight: 800; color: #3fb950; }
    .chem-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px; }
</style>
""", unsafe_allow_html=True)

st.title("🔥 Induced Thermal Runaway — Lead Time Analysis")
st.markdown(
    "Batch verification across battery abuse datasets. "
    "Measures the **anticipation lead time**: how many seconds before the 80°C threshold "
    "does the four-trigger engine first enter Watching Brief state?"
)

with st.expander("How the four-trigger engine works"):
    st.markdown("""
| Trigger | Mechanism | Battery physics |
|---------|-----------|-----------------|
| **A** | Statistical envelope (expanding ±3σ) | Temperature exits historical bounds |
| **B** | Rate density (dT/dt sustained > 0.5°C/s) | Persistent acceleration phase |
| **C** | 2nd derivative spike (d²T/dt² non-linear) | Inflection point before B can activate |
| **D** | Physics ODE residual (Newton's law) | Excess heat vs model — earliest signal |

Two or more triggers = **Watching Brief** → system alert issued.
""")

# ─── DATASET REGISTRY ───────────────────────────────────────────────────────
# Real paths (local dev / Google Drive). Fallback: synthetic samples from repo.

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "..", "data")
SNL_SYNTH = os.path.join(DATA_DIR, "snl_synthetic_samples.csv")

def _synth_for(dataset_id: str) -> pd.DataFrame | None:
    """Load one chemistry from the bundled synthetic SNL file."""
    if not os.path.exists(SNL_SYNTH):
        return None
    try:
        df = pd.read_csv(SNL_SYNTH)
        subset = df[df["dataset"] == dataset_id][["time","temperature","voltage","current"]].copy()
        return subset if len(subset) > 20 else None
    except Exception:
        return None

DATASETS = [
    {
        "label": "Synthetic Baseline (7-min lead-up)",
        "real_path": os.path.join(DATA_DIR, "thermal_runaway_data.csv"),
        "synth_id": None,
        "chemistry": "Synthetic",
    },
    {
        "label": "SNL NMC-LMO 85% SOC (Cell a)",
        "real_path": "/Users/edgemarsham/Library/CloudStorage/GoogleDrive-demarcusmuriel600@gmail.com/My Drive/Li_Battery_Project/data/processed/SNL_NMC-LMO_Graphite_26Ah_85SOC_a.csv",
        "synth_id": "SNL_NMC-LMO_85SOC_a",
        "chemistry": "NMC-LMO",
    },
    {
        "label": "SNL NMC-LMO 85% SOC (Cell b)",
        "real_path": "/Users/edgemarsham/Library/CloudStorage/GoogleDrive-demarcusmuriel600@gmail.com/My Drive/Li_Battery_Project/data/processed/SNL_NMC-LMO_Graphite_26Ah_85SOC_b.csv",
        "synth_id": "SNL_NMC-LMO_85SOC_b",
        "chemistry": "NMC-LMO",
    },
    {
        "label": "SNL LMO-LNO 50% SOC (Cell b)",
        "real_path": "/Users/edgemarsham/Library/CloudStorage/GoogleDrive-demarcusmuriel600@gmail.com/My Drive/Li_Battery_Project/data/processed/SNL_LMO-LNO_Graphite_33Ah_50SOC_b-copy.csv",
        "synth_id": "SNL_LMO-LNO_50SOC_b",
        "chemistry": "LMO-LNO",
    },
    {
        "label": "SNL LCO 90% SOC (Cell a)",
        "real_path": "/Users/edgemarsham/Library/CloudStorage/GoogleDrive-demarcusmuriel600@gmail.com/My Drive/Li_Battery_Project/data/processed/SNL_LCO_Graphite_6.4Ah_90SOC_a.csv",
        "synth_id": "SNL_LCO_90SOC_a",
        "chemistry": "LCO",
    },
]

COL_MAP = {
    "temp": "temperature", "t_c": "temperature", "t": "temperature",
    "timestamp": "time", "test_time (s)": "time", "temp_surface_1": "temperature",
}

def load_dataset(ds: dict) -> tuple[pd.DataFrame | None, str]:
    """Returns (df, source_label). Falls back to synthetic if real path unavailable."""
    # Try real path
    if os.path.exists(ds["real_path"]):
        try:
            df = pd.read_csv(ds["real_path"], comment="#")
            df.columns = df.columns.str.lower()
            for old, new in COL_MAP.items():
                if old in df.columns and new not in df.columns:
                    df.rename(columns={old: new}, inplace=True)
            if "time" not in df.columns:
                df["time"] = np.arange(len(df))
            if "temperature" not in df.columns:
                nums = df.select_dtypes(include=[np.number]).columns.tolist()
                df["temperature"] = df[nums[-1]] if nums else 0.0
            df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["time", "temperature"])
            return df, "real"
        except Exception:
            pass

    # Fallback: synthetic
    if ds["synth_id"]:
        df = _synth_for(ds["synth_id"])
        if df is not None:
            return df, "synthetic"

    # Last resort: bundled baseline
    baseline = os.path.join(DATA_DIR, "thermal_runaway_data.csv")
    if os.path.exists(baseline):
        try:
            df = pd.read_csv(baseline)
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            return df, "baseline_fallback"
        except Exception:
            pass

    return None, "unavailable"

# ─── BATCH ANALYSIS ─────────────────────────────────────────────────────────

RUNAWAY_TEMP = 80.0
results = []
charts = []
downloads = []

with st.spinner("Running four-trigger engine across all datasets …"):
    for ds in DATASETS:
        df_raw, source = load_dataset(ds)
        label = ds["label"]
        chem  = ds["chemistry"]

        if df_raw is None:
            results.append({"Dataset": label, "Chemistry": chem, "Source": "unavailable",
                             "T_max (°C)": "—", "System alert (s)": "—", "80°C alarm (s)": "—",
                             "Lead time (s)": "—", "Trig A": "—", "Trig B": "—", "Trig C": "—", "Trig D": "—"})
            continue

        try:
            df_c = run_hybrid_heuristic(df_raw, target_col="temperature")

            T_max   = df_c["temperature"].max()
            t_alarm = df_c.loc[df_c["temperature"] >= RUNAWAY_TEMP, "time"]
            t_alarm = float(t_alarm.iloc[0]) if not t_alarm.empty else None

            # System alert = first time 2+ triggers fire (Watching Brief)
            watching = df_c[df_c["trigger_sum"] >= 2]
            t_system = float(watching.iloc[0]["time"]) if not watching.empty else None

            lead = (t_alarm - t_system) if (t_alarm and t_system and t_system < t_alarm) else None

            trig_times = {}
            for letter in ["A", "B", "C", "D"]:
                col = f"trigger_{letter}"
                tp = df_c[df_c[col] == 1]
                trig_times[letter] = f"{tp.iloc[0]['time']:.0f}s" if not tp.empty else "—"

            source_label = {"real": "✓ real", "synthetic": "⚗ synthetic", "baseline_fallback": "⚗ fallback"}.get(source, source)
            results.append({
                "Dataset": label,
                "Chemistry": chem,
                "Source": source_label,
                "T_max (°C)": f"{T_max:.0f}",
                "System alert (s)": f"{t_system:.0f}" if t_system else "—",
                "80°C alarm (s)":   f"{t_alarm:.0f}"  if t_alarm  else "—",
                "Lead time (s)":    f"🔥 {lead:.0f}"   if lead     else ("reactive" if t_alarm else "—"),
                "Trig A": trig_times["A"],
                "Trig B": trig_times["B"],
                "Trig C": trig_times["C"],
                "Trig D": trig_times["D"],
            })

            downloads.append({"label": label[:12], "csv": df_c.to_csv(index=False).encode()})

            # Per-dataset chart
            COLORS = {"A": "#58a6ff", "B": "#d29922", "C": "#db6d28", "D": "#3fb950"}
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_c["time"], y=df_c["temperature"],
                                     name="Temp (°C)", line=dict(color="#ff7b72", width=2.5)))
            fig.add_trace(go.Scatter(x=df_c["time"], y=df_c["prob_tr"],
                                     name="State severity (%)", line=dict(color="#a371f7", width=1.5, dash="dot"),
                                     yaxis="y2"))
            for letter in ["A", "B", "C", "D"]:
                tp = df_c[df_c[f"trigger_{letter}"] == 1]
                if not tp.empty:
                    t0 = float(tp.iloc[0]["time"])
                    lead_str = f"{t_alarm - t0:.0f}s warning" if t_alarm and t0 < t_alarm else ""
                    fig.add_vline(x=t0, line_width=1.5, line_dash="dash", line_color=COLORS[letter],
                                  annotation_text=f"<b>{letter}</b> {lead_str}",
                                  annotation_font=dict(color=COLORS[letter], size=10),
                                  annotation_position="top left")
            if t_alarm:
                fig.add_vline(x=t_alarm, line_width=2, line_color="#ff7b72",
                              annotation_text="80°C alarm", annotation_font=dict(color="#ff7b72", size=11))
            if t_system and t_alarm and t_system < t_alarm:
                fig.add_vline(x=t_system, line_width=2, line_dash="dash", line_color="#3fb950",
                              annotation_text=f"System alert ({lead:.0f}s early)",
                              annotation_font=dict(color="#3fb950", size=11),
                              annotation_position="top right")

            title = f"{label} [{chem}]"
            if lead:
                title += f" — <b>{lead:.0f}s lead time</b>"
            fig.update_layout(
                template="plotly_dark", title=title, height=380,
                yaxis=dict(title="Temp (°C)"),
                yaxis2=dict(title="State %", overlaying="y", side="right", range=[0, 105]),
                margin=dict(l=10, r=10, t=50, b=30),
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            )
            charts.append(fig)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            results.append({"Dataset": label, "Chemistry": chem, "Source": source,
                             "T_max (°C)": f"Error: {e}", "System alert (s)": "—",
                             "80°C alarm (s)": "—", "Lead time (s)": "—",
                             "Trig A": "—", "Trig B": "—", "Trig C": "—", "Trig D": "—"})

# ─── HEADLINE SUMMARY ───────────────────────────────────────────────────────

st.markdown("### Lead-time summary")

valid = [r for r in results if r["Lead time (s)"] not in ("—", "reactive") and "🔥" in str(r["Lead time (s)"])]
if valid:
    leads = [float(r["Lead time (s)"].replace("🔥 ", "")) for r in valid]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="chem-label">Mean lead time</div><div class="lead-badge">{np.mean(leads):.0f}s</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="chem-label">Max lead time</div><div class="lead-badge">{max(leads):.0f}s</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="chem-label">Datasets detected</div><div class="lead-badge">{len(valid)} / {len(results)}</div></div>', unsafe_allow_html=True)

st.markdown("### Per-dataset results")
st.dataframe(pd.DataFrame(results), use_container_width=True)

if downloads:
    st.markdown("**Download analysed datasets:**")
    cols = st.columns(min(len(downloads), 5))
    for i, dl in enumerate(downloads):
        with cols[i % 5]:
            st.download_button(f"📥 {dl['label']}", data=dl["csv"],
                               file_name=f"analyzed_{dl['label'].replace(' ','_')}.csv",
                               mime="text/csv", key=f"dl_{i}")

st.markdown("---")
st.markdown("### Detection charts")
for fig in charts:
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "⚗ Datasets marked *synthetic* use physics-realistic traces generated from SNL chemistry profiles "
    "when the original files are not available in the deployment environment. "
    "Real SNL data will be used automatically when present."
)
