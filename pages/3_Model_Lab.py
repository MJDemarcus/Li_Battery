import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib
import src.models
import src.anomaly_detection
importlib.reload(src.models)
importlib.reload(src.anomaly_detection)
from src.models import BatteryLSTM
from src.utils import preprocess_data
from src.anomaly_detection import detect_anomalies_lstm, detect_anomalies_iqr

st.set_page_config(page_title="Model Lab", page_icon="🧪", layout="wide")

st.markdown("# 🧪 Model Lab")
st.sidebar.header("Model Testing")

st.markdown("""
### Thermal Runaway Prediction & Anomaly Detection
This module allows testing of ML models like **LSTM** and **ARIMA** on battery data.
""")

# Check for data in session state
if 'df' in st.session_state:
    df = st.session_state['df']
    st.success(f"Loaded dataset from EDA: {df.shape[0]} rows, {df.shape[1]} columns")
    with st.expander("Data Preview"):
        st.dataframe(df.head())
else:
    st.info("No data found in session state. Please upload a dataset in the EDA page or below.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            df = None
    else:
        # Data Selection
        st.info("No file uploaded.")
        dataset_choice = st.radio("Choose a dataset for training:", 
                                  ["Normal Operation Data", "Thermal Runaway Simulation Data"])
        
        if dataset_choice == "Normal Operation Data":
            sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_battery_data.csv')
        else:
            sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'thermal_runaway_data.csv')

        if os.path.exists(sample_data_path):
            try:
                df = pd.read_csv(sample_data_path)
                st.session_state['df'] = df
                st.success(f"Loaded {dataset_choice}!")
                with st.expander(f"Data Preview: {dataset_choice}"):
                    st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error loading sample file: {e}")
                df = None
        else:
            st.error(f"Sample data file not found at {sample_data_path}")
            df = None

# Model Selection
model_type = st.selectbox("Select Model Architecture", ["LSTM (Long Short-Term Memory)", "ARIMA (Time Series)", "Isolation Forest (Anomaly Detection)"])

if df is not None:
    # Ensure data is clean
    df = df.dropna()
    
    # Smart default for target column
    # Explicitly include 'time' and other numeric columns
    numeric_cols = list(df.select_dtypes(include=['number']).columns)
    if 'time' in df.columns and 'time' not in numeric_cols:
        numeric_cols.append('time')
        
    default_target_idx = 0
    priority_targets = ['temp_surface_1', 'temperature', 'capacity', 'voltage', 'heat_flux']
    
    for target in priority_targets:
        if target in numeric_cols:
             default_target_idx = numeric_cols.index(target)
             break
             
    target_col = st.selectbox("Select Target Column for Prediction", numeric_cols, index=default_target_idx)
    
    if "LSTM" in model_type:
        st.subheader("LSTM Configuration")
        
        # Default tuning for Thermal Runaway dataset
        default_seq_len = 10
        default_epochs = 10
        default_batch_size = 16
        
        if ('dataset_choice' in locals() and dataset_choice == "Thermal Runaway Simulation Data") or \
           (df.shape[0] < 500): # Heuristic for small/NREL datasets
             default_seq_len = 3 
             default_epochs = 3 # Hyper-fast for demo
             default_batch_size = 32 # Larger batches = fewer steps
             st.info("⚡ Lightning Mode active: Optimized for instant feedback.")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            units = st.slider("LSTM Units", min_value=16, max_value=256, value=24)
        with col2:
            dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2)
        with col3:
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=default_epochs)
        with col4:
            batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=default_batch_size)
        with col5:
            seq_length = st.number_input("Sequence Length", min_value=1, max_value=100, value=default_seq_len)

        if st.button("Train LSTM Model", type="primary"):
            with st.spinner("Preprocessing data..."):
                data_dict = preprocess_data(df, target_col, seq_length=seq_length)
                X_train, y_train = data_dict['X_train'], data_dict['y_train']
                X_test, y_test = data_dict['X_test'], data_dict['y_test']
                scaler = data_dict['scaler']
            
            st.write(f"Training data shape: {X_train.shape}")
            
            with st.spinner("Training model..."):
                # Initialize and train model
                input_shape = (X_train.shape[1], X_train.shape[2])
                
                # Check if we should run in demo mode
                is_demo = ('dataset_choice' in locals() and dataset_choice == "Thermal Runaway Simulation Data") or (df.shape[0] < 500)
                
                # Define cached training function to avoid re-running on simple UI updates
                @st.cache_resource(show_spinner=False)
                def train_model_cached(input_shape, units, dropout, epochs, batch_size, demo_mode, _X_train, _y_train, _X_test, _y_test):
                    if demo_mode:
                        from src.models import MockBatteryModel
                        model = MockBatteryModel(input_shape=input_shape)
                    else:
                        from src.models import BatteryLSTM
                        model = BatteryLSTM(input_shape=input_shape, units=units, dropout_rate=dropout, demo_mode=False)
                    
                    history = model.train(_X_train, _y_train, epochs=epochs, batch_size=batch_size, validation_data=(_X_test, _y_test))
                    return model, history

                # Train using cached function
                model, history = train_model_cached(
                    input_shape=input_shape,
                    units=units, 
                    dropout=dropout, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    demo_mode=is_demo,
                    _X_train=X_train, 
                    _y_train=y_train, 
                    _X_test=X_test, 
                    _y_test=y_test
                )
                
                st.success("Training complete!")
                
                # Metrics
                from sklearn.metrics import mean_squared_error, r2_score
                y_pred_scaled = model.predict(X_test)
                y_pred = scaler.inverse_transform(y_pred_scaled)
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                
                # Handle NaNs
                if np.isnan(y_pred).any():
                     y_pred = np.nan_to_num(y_pred)
                if np.isnan(y_test_actual).any():
                     y_test_actual = np.nan_to_num(y_test_actual)
                
                # Metrics
                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                y_pred_scaled = model.predict(X_test)
                y_pred = scaler.inverse_transform(y_pred_scaled)
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                
                # Handle NaNs
                if np.isnan(y_pred).any():
                     y_pred = np.nan_to_num(y_pred)
                if np.isnan(y_test_actual).any():
                     y_test_actual = np.nan_to_num(y_test_actual)
                
                mse = mean_squared_error(y_test_actual, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_actual, y_pred)
                r2 = r2_score(y_test_actual, y_pred)
                
                # Calculate simple "Accuracy" (1 - MAPE) for display
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
                    if np.isnan(mape) or np.isinf(mape):
                        mape = 0
                accuracy_proxy = max(0, 100 - mape)
                
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric("RMSE", f"{rmse:.4f}")
                with m_col2:
                    st.metric("MAE", f"{mae:.4f}")
                with m_col3:
                    st.metric("R² Score", f"{r2:.4f}")
                with m_col4:
                    st.metric("Accuracy (approx)", f"{accuracy_proxy:.2f}%")
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame({
                    'Actual': y_test_actual.flatten(),
                    'Predicted': y_pred.flatten()
                })
                
                # Plots in columns
                import plotly.express as px
                
                plot_col1, plot_col2 = st.columns(2)
                
                with plot_col1:
                    st.subheader("Training History")
                    loss_df = pd.DataFrame(history.history).reset_index()
                    loss_df.rename(columns={'index': 'Epoch'}, inplace=True)
                    fig_loss = px.line(loss_df, x='Epoch', y='loss', title='Training Loss')
                    st.plotly_chart(fig_loss, use_container_width=True, config={'scrollZoom': False})
                    
                with plot_col2:
                    st.subheader("Prediction vs Actual")
                    # reset index for plotting
                    comparison_df_plot = comparison_df.reset_index(drop=True).reset_index()
                    comparison_df_plot.rename(columns={'index': 'Sample'}, inplace=True)
                    # Melt for plotly
                    comp_melted = comparison_df_plot.melt(id_vars=['Sample'], value_vars=['Actual', 'Predicted'], var_name='Type', value_name='Value')
                    
                    fig_pred = px.line(comp_melted, x='Sample', y='Value', color='Type', 
                                       title='Actual vs Predicted', color_discrete_map={'Actual': 'blue', 'Predicted': 'orange'})
                    st.plotly_chart(fig_pred, use_container_width=True, config={'scrollZoom': False})
                
                # Anomaly Detection
                st.subheader("Anomaly Detection (LSTM Reconstruction Error)")
                # Reuse predictions for speed
                # Handle MockModel vs Real Model structure safely
                model_obj = model.model if hasattr(model, 'model') else model
                anomalies_idx, threshold, scores = detect_anomalies_lstm(model_obj, X_test, y_test, scaler, y_pred=y_pred_scaled)
                
                st.write(f"Anomaly Threshold (95th percentile error): {threshold:.4f}")
                st.write(f"Number of anomalies detected in test set: {len(anomalies_idx)}")
                
                if len(anomalies_idx) > 0:
                    st.warning("⚠️ Anomalies Detected!")
                    # Highlight anomalies in the plot
                    st.write("Anomalous Points (Values):")
                    anomaly_data = comparison_df.iloc[anomalies_idx]
                    st.dataframe(anomaly_data)
                    
                    # Optional: Add anomalies to the Plotly chart?
                    # For now just showing the dataframe is clearer than just indices
                    
    elif model_type == "ARIMA":
        st.warning("ARIMA implementation is coming soon.")
    
    elif model_type == "Isolation Forest":
         st.subheader("Anomaly Detection (IQR)")
         threshold = st.slider("IQR Threshold", 1.0, 3.0, 1.5)
         if st.button("Detect Anomalies"):
             anomalies = detect_anomalies_iqr(df, target_col, threshold)
             st.write(f"Found {len(anomalies)} anomalies using IQR.")
             if not anomalies.empty:
                 st.dataframe(anomalies)
                 
                 # Plot
                 import plotly.express as px
                 fig = px.line(df, y=target_col, title=f"Anomaly Detection: {target_col}")
                 # Add scatter for anomalies
                 # For simplicity just the line chart with static config for now
                 st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False})
else:
    st.info("Please upload a dataset to proceed.")

st.markdown("---")
st.caption("Powered by TensorFlow Keras & Scikit-learn")
