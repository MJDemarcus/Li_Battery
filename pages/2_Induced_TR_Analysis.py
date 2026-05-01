import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hybrid_evaluator import run_hybrid_heuristic

st.set_page_config(page_title="Induced TR Analysis", page_icon="🔥", layout="wide")

st.markdown("""
<style>
    .metric-box { background: rgba(33, 38, 45, 0.85); border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }
    .critical-value { color: #ff7b72; font-weight: bold; }
    .stApp { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)

st.title("🔥 Induced Thermal Runaway Anticipation Analysis")
st.markdown("Automated batch verification across specific 100% SOC (State of Charge) induced runaway datasets to measure the **exact mathematical anticipation lead time** using the A/B/C heuristic.")

with st.expander("ℹ️ How Does the A-B-C Heuristic Work?"):
    st.markdown("""
    **A. Prophet Expected Range (Mod 1)**: Prevent normal-heat false alarms using a moving baseline limit.
    **B. CNN-LSTM Surrogate (Mod 2)**: Sustained kinetic heating density ($dT/dt > 0.5$).
    **C. Kinetic Spike Detection (Mod 3)**: Second derivative exponential blowout ($e^{d^2T/dt^2}$).
    """)

datasets = {
    "Synthetic Baseline (7m Lead-Up)": os.path.join(os.path.dirname(__file__), '..', 'data', 'thermal_runaway_data.csv'),
    "SNL NMC-LMO 85% SOC (Cell b)": "/Users/edgemarsham/Library/CloudStorage/GoogleDrive-demarcusmuriel600@gmail.com/My Drive/Li_Battery_Project/data/processed/SNL_NMC-LMO_Graphite_26Ah_85SOC_b.csv",
    "SNL NMC-LMO 85% SOC (Cell a)": "/Users/edgemarsham/Library/CloudStorage/GoogleDrive-demarcusmuriel600@gmail.com/My Drive/Li_Battery_Project/data/processed/SNL_NMC-LMO_Graphite_26Ah_85SOC_a.csv",
    "SNL LMO-LNO 50% SOC (Cell b)": "/Users/edgemarsham/Library/CloudStorage/GoogleDrive-demarcusmuriel600@gmail.com/My Drive/Li_Battery_Project/data/processed/SNL_LMO-LNO_Graphite_33Ah_50SOC_b-copy.csv",
    "SNL LCO 90% SOC (Cell a)": "/Users/edgemarsham/Library/CloudStorage/GoogleDrive-demarcusmuriel600@gmail.com/My Drive/Li_Battery_Project/data/processed/SNL_LCO_Graphite_6.4Ah_90SOC_a.csv"
}

results = []
figs = []
downloads = []

with st.spinner("Processing 5 Induced TR Simulation Datasets..."):
    for name, path in datasets.items():
        try:
            if not os.path.exists(path):
                results.append({"Dataset": name, "Temp Range (C)": "N/A", "Runaway Extent (T_max)": "N/A", "Earliest Warning [3/3 Triggers]": "N/A", "Anticipation Lead Time": "N/A"})
                continue
                
            df_raw = pd.read_csv(path, comment='#')
            df_raw.columns = df_raw.columns.str.lower()
            
            # Map columns
            col_map = {'temp': 'temperature', 't_c': 'temperature', 't': 'temperature', 'timestamp': 'time', 'test_time (s)': 'time'}
            for old, new in col_map.items():
                if old in df_raw.columns and new not in df_raw.columns:
                    df_raw.rename(columns={old: new}, inplace=True)
            
            if 'time' not in df_raw.columns:
                df_raw['time'] = np.arange(len(df_raw))
            if 'temperature' not in df_raw.columns:
                num_c = df_raw.select_dtypes(include=[np.number]).columns.tolist()
                df_raw['temperature'] = df_raw[num_c[-1]] if num_c else np.zeros(len(df_raw))
                
            df_raw = df_raw.dropna(subset=['time', 'temperature'])
            
            # Apply Hybrid Evaluator
            df_computed = run_hybrid_heuristic(df_raw, target_col='temperature', k_spike=5.0)
            
            T_max = df_computed['temperature'].max()
            T_min = df_computed['temperature'].min()
            
            # Failure is exactly T_max (sensor dropout / explosion)
            t_failure = df_computed.loc[df_computed['temperature'] == T_max, 'time'].iloc[0]
            
            # Warning is the earliest moment ANY state trigger activates (>20%)
            warning_points = df_computed[df_computed['prob_tr'] > 20.0]
            t_warning = warning_points.iloc[0]['time'] if not warning_points.empty else None
            
            max_prob = warning_points['prob_tr'].max() if not warning_points.empty else 0
            if max_prob > 90: max_state_str = "4/4"
            elif max_prob > 70: max_state_str = "3/4"
            elif max_prob > 45: max_state_str = "2/4"
            else: max_state_str = "1/4"
            
            lead_time = (t_failure - t_warning) if t_warning is not None else None
            
            trig_times = {}
            for t_let in ["A", "B", "C", "D"]:
                tp = df_computed[df_computed[f"trigger_{t_let}"] == 1]
                trig_times[t_let] = f"{tp.iloc[0]['time']:.1f}s" if not tp.empty else "-"
            
            results.append({
                "Dataset": name,
                "Runaway Extent (T_max)": f"T={t_failure:.1f}s",
                "Earliest System Warning": f"T={t_warning:.1f}s (Max {max_state_str})" if t_warning else "No Warning",
                "Anticipation Lead Time": f"🔥 {lead_time:.1f}s" if lead_time is not None and lead_time > 0 else ("Lag/Reactive" if lead_time is not None else "N/A"),
                "Trig A": trig_times["A"],
                "Trig B": trig_times["B"],
                "Trig C": trig_times["C"],
                "Trig D": trig_times["D"]
            })
            
            downloads.append({"name": name, "data": df_computed.to_csv(index=False).encode('utf-8')})
            
            # Subplot - Remove ALL data after the blast/failure mark
            idx_limit = t_failure if t_failure else df_computed['time'].max()
            df_plot = df_computed[df_computed['time'] <= idx_limit]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['temperature'], name="Temp (°C)", line=dict(color="#ff7b72", width=3)))
            fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['prob_tr'], name="State Severity %", line=dict(color="#a371f7", width=2, dash='dot'), yaxis="y2"))
            
            # Annotate individual triggers AND calculate Time to Runaway
            colors = {"A": "#58a6ff", "B": "#d29922", "C": "#db6d28", "D": "#3fb950"}
            for trig_letter in ["A", "B", "C", "D"]:
                trig_col = f"trigger_{trig_letter}"
                trig_points = df_plot[df_plot[trig_col] == 1]
                if not trig_points.empty:
                    t_first = trig_points.iloc[0]['time']
                    t_to_runaway = (t_failure - t_first) if t_failure else 0
                    
                    # Create a visually rich annotation label
                    label = f"<b>{trig_letter}</b><br>{t_to_runaway:.1f}s warning" if t_to_runaway > 0 else f"<b>{trig_letter}</b>"
                    fig.add_vline(x=t_first, line_width=1.5, line_dash="dash", line_color=colors[trig_letter], 
                                  annotation_text=label, annotation_position="top left", annotation_font=dict(color=colors[trig_letter], size=10))
            
            if t_failure:
                fig.add_vline(x=t_failure, line_width=2, line_dash="solid", line_color="#ff7b72", annotation_text="💥 Runaway", annotation_font=dict(size=14, color="#ff7b72"))
                
            fig.update_layout(
                title=f"Anticipation Window: {name}",
                yaxis=dict(title="Temp (°C)"),
                yaxis2=dict(title="State Triggers (%)", overlaying="y", side="right", range=[0, 105]),
                margin=dict(l=20, r=20, t=40, b=20),
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            figs.append(fig)
            
        except Exception as dummy_e:
            import traceback
            err_details = traceback.format_exc()
            print(f"Error on {name}:\n{err_details}")
            results.append({"Dataset": name, "Runaway Extent (T_max)": f"Error: {dummy_e}", "Earliest System Warning": "N/A", "Anticipation Lead Time": "N/A", "Trig A": "-", "Trig B": "-", "Trig C": "-", "Trig D": "-"})

st.markdown("### Aggregated Lead Time Verification")
st.table(pd.DataFrame(results))

# Show quick download links safely
if len(downloads) > 0:
    st.markdown("**Download Analyzed Datasets:**")
    cols = st.columns(len(downloads))
    for idx, dl in enumerate(downloads):
        with cols[idx]:
            st.download_button(label=f"📥 {dl['name'][:10]}...", data=dl['data'], file_name=f"annotated_{dl['name'].replace(' ', '_')}.csv", mime="text/csv", key=f"dl_{idx}")

st.write("---")
st.markdown("### Sequential Runaway Charts")
for plot in figs:
    st.plotly_chart(plot, use_container_width=True)
