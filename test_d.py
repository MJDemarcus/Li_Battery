import pandas as pd
from src.hybrid_evaluator import run_hybrid_heuristic
df = pd.read_csv('data/thermal_runaway_data.csv')
df.rename(columns={'timestamp': 'time'}, inplace=True)
df_c = run_hybrid_heuristic(df)
print(df_c['trigger_D'].value_counts())
