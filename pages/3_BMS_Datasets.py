import streamlit as st
import os

st.set_page_config(page_title="BMS Datasets", page_icon="🗃️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    header {visibility: hidden;}
    
    h1, h2, h3, h4 {
        color: #58a6ff !important;
        font-weight: 600;
    }
    
    .dataset-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease-in-out, border-color 0.2s ease-in-out;
    }
    
    .dataset-card:hover {
        transform: translateY(-4px);
        border-color: #58a6ff;
    }
    
    .dataset-title {
        color: #58a6ff;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .dataset-info {
        font-size: 0.95rem;
        color: #8b949e;
        margin-bottom: 8px;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-normal {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
    }
    
    .badge-tr {
        background-color: rgba(248, 81, 73, 0.15);
        color: #ff7b72;
        border: 1px solid rgba(248, 81, 73, 0.4);
    }
    
    .location {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        background: rgba(110, 118, 129, 0.1);
        padding: 3px 6px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #e6edf3;
    }
    
    .description-text {
        margin-top: 16px;
        color: #c9d1d9;
        font-size: 0.95rem;
        line-height: 1.6;
        border-top: 1px solid #30363d;
        padding-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🗃️ BMS Datasets Catalog")
st.markdown("A comprehensive catalog of Battery Management System (BMS) datasets sourced from the web and stored in our project repository and Google Drive. These datasets are crucial for identifying baseline behaviors and training our predictive anomaly detection models.")

datasets = [
    {
        "name": "Battery Archive - Normal Cycling",
        "filename": "battery_archive_cycling.csv",
        "type": "Normal",
        "description": "Baseline cycling data demonstrating standard charge and discharge dynamics for healthy cells. Useful for establishing the Prophet expectation bounds.",
        "location": "battery_safety_platform/data/external/...",
        "web": "Originally sourced from BatteryArchive.org"
    },
    {
        "name": "NASA - Baseline Aging Sample",
        "filename": "nasa_aging_sample.csv",
        "type": "Normal",
        "description": "Long-term capacity degradation dataset. Shows how voltage and capacity drift safely over hundreds of normal cycles.",
        "location": "battery_safety_platform/data/external/...",
        "web": "Sourced from NASA Prognostics Center of Excellence"
    },
    {
        "name": "Sample Battery Data",
        "filename": "sample_battery_data.csv",
        "type": "Normal",
        "description": "Clean, synthesized and normalized metric logs. Perfect for isolated testing of the Continuous Monitoring MVP.",
        "location": "battery_safety_platform/data/...",
        "web": "Internal project generation / benchmark"
    },
    {
        "name": "NREL Abuse Test Sample",
        "filename": "nrel_abuse_test_sample.csv",
        "type": "Forced TR",
        "description": "Nail-penetration and extreme heat abuse variables. Demonstrates rapid onset of Thermal Runaway through physical damage and extreme duress.",
        "location": "battery_safety_platform/data/external/...",
        "web": "Sourced from National Renewable Energy Laboratory (NREL)"
    },
    {
        "name": "NREL Battery Failure Databank",
        "filename": "battery-failure-databank-revision2-feb24.xlsx",
        "type": "Forced TR",
        "description": "Extensive multi-cell testing logs containing destructive failure and gas emission profiles. Critical for validating Phase-3 runaway detection boundaries.",
        "location": "Google Drive/My Drive/Li_Battery_Project/data/...",
        "web": "Sourced from NREL Databank"
    },
    {
        "name": "NASA TR Databank",
        "filename": "nasa_tr_databank.xlsx",
        "type": "Forced TR",
        "description": "Accelerated calorimetry measurements capturing catastrophic phase-change temperatures and instantaneous voltage drops.",
        "location": "Google Drive/My Drive/Li_Battery_Project/data/...",
        "web": "Sourced from NASA Open Data"
    },
    {
        "name": "Mendeley TR Data",
        "filename": "mendeley_tr_data.zip",
        "type": "Forced TR",
        "description": "Large-scale thermal runaway test compendium containing both electrical waveforms and extreme environmental temperature triggers.",
        "location": "Google Drive/My Drive/Li_Battery_Project/data/...",
        "web": "Sourced from Mendeley Data"
    },
    {
        "name": "Thermal Runaway Simulation Data",
        "filename": "thermal_runaway_data.csv",
        "type": "Forced TR",
        "description": "Highly condensed representation of thermal escalation used for rapid validation of the Second-Derivative thresholding.",
        "location": "battery_safety_platform/data/...",
        "web": "Internal project benchmark"
    }
]

st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2 style='text-align: center; color: #3fb950 !important; font-size: 1.8rem;'><span style='margin-right:8px;'>✅</span>Normal Operation</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e; margin-bottom: 2rem;'>Datasets capturing safe, expected battery behavior including regular cycling and nominal aging drift.</p>", unsafe_allow_html=True)
    
    for ds in datasets:
        if ds["type"] == "Normal":
            st.markdown(f"""
            <div class="dataset-card">
                <div class="dataset-title">
                    <span class="badge badge-normal">Normal</span> {ds['name']}
                </div>
                <div class="dataset-info"><strong>Filename:</strong> <span class="location">{ds['filename']}</span></div>
                <div class="dataset-info"><strong>Origin:</strong> {ds['web']}</div>
                <div class="dataset-info"><strong>Location:</strong> <span class="location">{ds['location']}</span></div>
                <div class="description-text">{ds['description']}</div>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown("<h2 style='text-align: center; color: #ff7b72 !important; font-size: 1.8rem;'><span style='margin-right:8px;'>🔥</span>Forced Thermal Runaway</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e; margin-bottom: 2rem;'>Destructive abuse tests, extreme heating arrays, and short-circuit experiments pushing cells past critical limits.</p>", unsafe_allow_html=True)
    
    for ds in datasets:
        if ds["type"] == "Forced TR":
            st.markdown(f"""
            <div class="dataset-card">
                <div class="dataset-title">
                    <span class="badge badge-tr">Forced TR</span> {ds['name']}
                </div>
                <div class="dataset-info"><strong>Filename:</strong> <span class="location">{ds['filename']}</span></div>
                <div class="dataset-info"><strong>Origin:</strong> {ds['web']}</div>
                <div class="dataset-info"><strong>Location:</strong> <span class="location">{ds['location']}</span></div>
                <div class="description-text">{ds['description']}</div>
            </div>
            """, unsafe_allow_html=True)
