import streamlit as st

st.set_page_config(
    page_title="Battery Safety Platform",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔋 Battery Safety Platform")

st.markdown("""
## Welcome to the Battery Safety Platform

This platform is dedicated to advancing lithium-ion battery safety through:
1.  **Repository of Public Information**: Curated research and datasets.
2.  **Exploratory Data Analysis (EDA)**: Tools to visualize battery behavior.
3.  **Model Testing**: Predictive modelling for thermal runaway and anomaly detection.

### Getting Started
Use the sidebar to navigate to different modules:
- **Knowledge Base**: Learn about the latest safety methodologies.
- **EDA**: Upload and analyze your battery datasets.
- **Model Lab**: Test predictive models for anomaly detection.

### Key Objectives
- **Acoustic Monitoring**: Detecting early failure sounds.
- **Intelligent BMS**: Real-time health monitoring.
- **Thermal Runaway Prediction**: Modelling critical failure events.

---
*Built for research and development in battery safety.*
""")

st.sidebar.success("Select a module above.")
