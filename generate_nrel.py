import pandas as pd
import numpy as np

# Simulate a thermal runaway event over 300 seconds
time = np.arange(0, 301, 1) # 301 points
voltage = np.ones(len(time)) * 4.2
temp_1 = np.ones(len(time)) * 25.0
temp_2 = np.ones(len(time)) * 25.0
heat_flux = np.zeros(len(time))
pressure = np.zeros(len(time))

# Event dynamics
runaway_start = 150
peak = 240

for i, t in enumerate(time):
    if t < runaway_start:
        # Slow heating phase
        temp_1[i] += t * 0.1
        temp_2[i] += t * 0.08
        voltage[i] -= t * 0.0005
    elif t < peak:
        # Runaway acceleration
        dt = t - runaway_start
        temp_1[i] = temp_1[runaway_start] + dt**1.8
        temp_2[i] = temp_2[runaway_start] + dt**1.7
        voltage[i] = max(0, 4.2 - np.exp(dt * 0.05))
        heat_flux[i] = dt * 50
        pressure[i] = dt * 0.5
    else:
        # Cooling / Post-event
        dt = t - peak
        temp_1[i] = max(25, 450 - dt * 2)
        temp_2[i] = max(25, 430 - dt * 2)
        voltage[i] = 0
        heat_flux[i] = max(0, 5000 - dt * 100)
        pressure[i] = max(0, 45 - dt * 0.5)

df = pd.DataFrame({
    'time_sec': time,
    'voltage': voltage,
    'temp_surface_1': temp_1,
    'temp_surface_2': temp_2,
    'heat_flux': heat_flux,
    'vent_gas_pressure': pressure
})

output_path = 'data/external/nrel_abuse_test_sample.csv'
df.to_csv(output_path, index=False)
print(f"Generated {len(df)} rows to {output_path}")
