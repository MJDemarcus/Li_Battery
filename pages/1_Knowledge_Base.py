import streamlit as st
import os

st.set_page_config(page_title="Knowledge Base", page_icon="📚")

st.markdown("# 📚 Knowledge Base")
st.sidebar.header("Knowledge Base")

# Function to load markdown content
def load_markdown(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read()
    return None

# Display Summary
st.markdown("## Battery Safety Research Key Areas")

with st.expander("🔊 Acoustic Monitoring (\"Hearing\" Failures)", expanded=True):
    st.markdown("""
    **Core Concept:** Detecting the distinct sound of a lithium-ion battery's safety valve breaking, which often occurs just before ignition.
    
    *   **Method:** Detecting the "click-hiss" sound of venting gases (usually 2-3 minutes before fire).
    *   **Performance:** Models can achieve ~94% accuracy, distinguishing this sound from background noise.
    *   **Application:** Early warning systems in EVs, parking garages, and warehouses.
    *   **Reference:** Research by NIST and others.
    """)

with st.expander("🧠 Intelligent Battery Management Systems (BMS)", expanded=True):
    st.markdown("""
    **Core Concept:** Integration of ML into BMS to monitor battery health in real-time.
    
    *   **Anomaly Detection:** Analyzing sensor data (voltage, current, temperature) to find unusual patterns indicating cell imbalance or impending thermal runaway.
    *   **Predictive Maintenance:** Identifying "hidden" faults or degradation using historical data.
    *   **Early Warning Signs:** Detecting subtle temperature fluctuations or pressure changes.
    """)

with st.expander("🔥 Thermal Runaway Prediction and Modelling", expanded=True):
    st.markdown("""
    **Core Concept:** Using ML to predict the behavior of a battery pack during a thermal runaway event.
    
    *   **Goal:** Essential for designing safer batteries and containment systems.
    *   **Modelling:** Simulating heat propagation and failure modes.
    """)

st.markdown("---")
st.markdown("## 🌍 Open Battery Data Repositories")
st.info("The following repositories provided by major institutions are the gold standard for battery research. We have included **sample snapshots** of these in the **EDA** tab for you to explore immediately.")

with st.expander("🏛️ NREL Battery Failure Databank (National Renewable Energy Laboratory)"):
    st.markdown("""
    **Focus:** Thermal runaway and abuse testing (nail penetration, overheating).
    *   **Data Types:** Heat flux, gas pressure, temperature maps during explosion events.
    *   **Use Case:** Benchmarking safety risks and designing containment systems.
    *   **Link:** [NREL Battery Failure Databank](https://www.nrel.gov/transportation/battery-failure.html)
    """)

with st.expander("🚀 NASA Li-ion Battery Aging Datasets"):
    st.markdown("""
    **Focus:** Run-to-failure cycling and prognostics.
    *   **Data Types:** Voltage, current, temperature, and capacity fade over hundreds of cycles.
    *   **Use Case:** Predicting Remaining Useful Life (RUL).
    *   **Link:** [NASA Prognostics Center](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
    """)

with st.expander("🔋 Battery Archive (Sandia, Oxford, etc.)"):
    st.markdown("""
    **Focus:** Multi-institution cycling data under various conditions.
    *   **Data Types:** Standardized cycle aging data (efficiency, discharge energy) from diverse labs.
    *   **Use Case:** Comparing degradation across different cell chemistries and environments.
    *   **Link:** [Battery Archive](https://www.batteryarchive.org/)
    """)

with st.expander("⚡ EPRI BESS Failure Incident Database"):
    st.markdown("""
    **Focus:** Real-world failure incidents of Battery Energy Storage Systems.
    *   **Data Types:** Incident reports, root cause analyses of fires/explosions.
    *   **Link:** [EPRI Storage Wiki](https://storagewiki.epri.com/index.php/BESS_Failure_Incident_Database)
    """)

with st.expander("🔥 UL Solutions Incident Reporting"):
    st.markdown("""
    **Focus:** tracking thermal runaway events in consumer electronics and EVs.
    *   **Link:** [UL Solutions](https://www.ul.com/insights/lithium-ion-battery-incident-reporting)
    """)

st.markdown("---")
st.markdown("### Project README")

# We can display the README content here or specific docs
readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")
readme_content = load_markdown(readme_path)

if readme_content:
    with st.expander("View Full Project README"):
        st.markdown(readme_content)
else:
    st.warning("README.md not found.")
