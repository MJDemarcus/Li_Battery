import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

def train_and_plot(df, column_name, title, ylabel, filename):
    print(f"Fitting Prophet model for {column_name}...")
    
    # Prepare data for Prophet
    model_df = df[['ds', column_name]].rename(columns={column_name: 'y'})
    
    # Initialize and fit model
    model = Prophet(
        changepoint_prior_scale=0.05,
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False
    )
    model.fit(model_df)
    
    # Make future predictions (50 seconds ahead)
    future = model.make_future_dataframe(periods=50, freq='s')
    forecast = model.predict(future)
    
    # Visualize
    print(f"Generating visualization for {column_name}...")
    fig = model.plot(forecast)
    plt.title(title)
    plt.xlabel('Time (Mapped to Datetime)')
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")
    
    # Simple MAE
    df_merged = model_df.merge(forecast[['ds', 'yhat']], on='ds')
    mae = (df_merged['y'] - df_merged['yhat']).abs().mean()
    print(f"{column_name} Mean Absolute Error: {mae:.4f}")
    return mae

def run_baselines():
    # 1. Load Data
    data_path = 'data/sample_battery_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    print("Data loaded successfully.")

    # 2. Preprocess Time
    start_date = datetime(2024, 1, 1)
    df['ds'] = df['time'].apply(lambda x: start_date + timedelta(seconds=x))

    # 3. Generate Baselines
    metrics = [
        ('capacity', 'Li Battery Capacity Baseline (Prophet)', 'Capacity (%)', 'baseline_capacity.png'),
        ('voltage', 'Li Battery Voltage Baseline (Prophet)', 'Voltage (V)', 'baseline_voltage.png'),
        ('temperature', 'Li Battery Temperature Baseline (Prophet)', 'Temperature (C)', 'baseline_temperature.png')
    ]

    for col, title, ylabel, fname in metrics:
        train_and_plot(df, col, title, ylabel, fname)

if __name__ == "__main__":
    run_baselines()
