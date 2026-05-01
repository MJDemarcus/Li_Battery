import os
import numpy as np
import pandas as pd

def generate_noise(n, scale=0.3):
    return np.random.normal(0, scale, n)

def save_ds(name, t, temp, out_dir):
    df = pd.DataFrame({"time": t, "temperature": np.clip(temp, a_min=-20, a_max=600)})
    df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    print(f"Generated: {name}.csv")

def generate_datasets():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bms_simulations")
    os.makedirs(out_dir, exist_ok=True)
    
    np.random.seed(42)
    
    # --- NORMAL DATASETS ---
    t = np.arange(0, 1000, 1)
    
    # 1. Highway Driving
    temp = 25 + 15 * np.sin(t / 200) + generate_noise(len(t))
    save_ds("Normal_01_Highway_Driving", t, temp, out_dir)
    
    # 2. Fast Charging
    temp = 25 + 30 * (1 - np.exp(-t / 150))
    temp[500:] = temp[500] * np.exp(-(t[500:] - 500) / 200) # Cooling
    temp += generate_noise(len(t), 0.2)
    save_ds("Normal_02_Fast_Charging", t, temp, out_dir)
    
    # 3. City Commute
    temp = 30 + 5 * np.sin(t / 50) + 3 * np.cos(t / 10) + generate_noise(len(t), 0.5)
    save_ds("Normal_03_City_Commute", t, temp, out_dir)
    
    # 4. High Temp Storage
    temp = 60 + generate_noise(len(t), 0.1)
    save_ds("Normal_04_High_Temp_Storage", t, temp, out_dir)
    
    # 5. Cold Start
    temp = -5 + 35 * (1 - np.exp(-t / 300)) + generate_noise(len(t), 0.4)
    save_ds("Normal_05_Cold_Start", t, temp, out_dir)
    
    # --- THERMAL RUNAWAY DATASETS ---
    # 1. Nail Penetration (Instant blast)
    temp = 25 + generate_noise(len(t))
    temp += 10 * np.exp((t - 600) / 25) * (t > 600)
    save_ds("Runaway_01_Nail_Penetration", t, temp, out_dir)
    
    # 2. Slow Overcharge
    temp = 25 + 20 * (t / 1000) + generate_noise(len(t), 0.2)
    temp += 5 * np.exp((t - 700) / 40) * (t > 700)
    save_ds("Runaway_02_Slow_Overcharge", t, temp, out_dir)
    
    # 3. External Short
    temp = 25 + 50 * (1 - np.exp(-t / 100)) + generate_noise(len(t), 0.6)
    temp += 8 * np.exp((t - 550) / 30) * (t > 550)
    save_ds("Runaway_03_External_Short", t, temp, out_dir)
    
    # 4. Crush Damage
    temp = 30 + generate_noise(len(t), 0.3)
    temp += 6 * np.exp((t - 450) / 35) * (t > 450)
    save_ds("Runaway_04_Crush_Damage", t, temp, out_dir)
    
    # 5. Internal Separator Failure (Very slow creep)
    temp = 35 + generate_noise(len(t), 0.2)
    temp += 2 * np.exp((t - 300) / 60) * (t > 300)
    save_ds("Runaway_05_Internal_Separator_Failure", t, temp, out_dir)

if __name__ == "__main__":
    generate_datasets()
