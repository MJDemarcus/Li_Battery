import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📊")

st.markdown("# 📊 Exploratory Data Analysis (EDA)")
st.sidebar.header("EDA Tools")

st.write("Upload your battery dataset (CSV) to perform initial analysis.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df  # Save to session state
        st.success("File uploaded successfully and saved to session state!")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Dataset Statistics")
        st.write(df.describe())
        
        # Column selection for plotting
        st.subheader("Visualization")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if numeric_columns:
            x_axis = st.selectbox("Select X-axis", options=df.columns, index=0)
            y_axis = st.multiselect("Select Y-axis (one or more)", options=numeric_columns, default=[numeric_columns[0]])
            
            if y_axis:
                fig = px.line(df, x=x_axis, y=y_axis, title="Time Series Plot")
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation Matrix
                if st.checkbox("Show Correlation Matrix"):
                    st.subheader("Correlation Matrix")
                    corr = df[numeric_columns].corr()
                    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                    st.plotly_chart(fig_corr)
        else:
            st.warning("No numeric columns found for visualization.")
            
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    # Data Selection
    st.info("No file uploaded.")
    
    data_source = st.radio("Select Data Source:", ["Platform Samples", "Public Datasets Repository"])
    
    sample_data_path = None
    dataset_name = ""
    
    if data_source == "Platform Samples":
        dataset_choice = st.selectbox("Choose a sample dataset:", 
                                  ["Normal Operation Data", "Thermal Runaway Simulation Data"])
        if dataset_choice == "Normal Operation Data":
            sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_battery_data.csv')
            dataset_name = "Normal Operation"
        else:
            sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'thermal_runaway_data.csv')
            dataset_name = "Thermal Runaway"
            
    else:
        dataset_choice = st.selectbox("Choose a public repository sample:", 
                                      ["NASA Aging Dataset (Cycle Life)", 
                                       "NREL Battery Failure (Abuse Test)", 
                                       "Battery Archive (Cycling Comparison)"])
        
        if dataset_choice == "NASA Aging Dataset (Cycle Life)":
            sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'external', 'nasa_aging_sample.csv')
            dataset_name = "NASA Aging"
        elif dataset_choice == "NREL Battery Failure (Abuse Test)":
            sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'external', 'nrel_abuse_test_sample.csv')
            dataset_name = "NREL Abuse"
        else:
            sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'external', 'battery_archive_cycling.csv')
            dataset_name = "Battery Archive"

    if os.path.exists(sample_data_path):
        df = pd.read_csv(sample_data_path)
        st.session_state['df'] = df
        st.success(f"Loaded {dataset_name}!")
        st.subheader(f"Data Preview: {dataset_name}")
        st.dataframe(df.head())
        
        # Proceed with visualization logic for sample data
        st.subheader("Dataset Statistics")
        st.write(df.describe())
        
        st.subheader("Visualization")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if numeric_columns:
            x_axis = st.selectbox("Select X-axis", options=df.columns, index=0)
            
            # Smart Default Y-axis based on dataset type
            default_y = [numeric_columns[1]] if len(numeric_columns) > 1 else []
            
            if dataset_name == "Thermal Runaway" and "temperature" in df.columns:
                 default_y = ["temperature", "voltage"]
            elif dataset_name == "NASA Aging" and "capacity" in df.columns:
                default_y = ["capacity"]
            elif dataset_name == "NREL Abuse":
                possible_cols = ["heat_flux", "vent_gas_pressure", "temp_surface_1"]
                default_y = [col for col in possible_cols if col in df.columns]
            elif dataset_name == "Battery Archive" and "efficiency" in df.columns:
                default_y = ["efficiency", "discharge_energy"]
            
            y_axis = st.multiselect("Select Y-axis (one or more)", options=numeric_columns, default=default_y)
            
            if y_axis:
                fig = px.line(df, x=x_axis, y=y_axis, title=f"Trends in {dataset_name}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation Matrix
                if st.checkbox("Show Correlation Matrix"):
                    st.subheader("Correlation Matrix")
                    corr = df[numeric_columns].corr()
                    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                    st.plotly_chart(fig_corr)
    else:
         st.error(f"Sample data file not found at {sample_data_path}")
