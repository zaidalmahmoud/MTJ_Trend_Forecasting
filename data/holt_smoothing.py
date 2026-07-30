import pandas as pd
import numpy as np
import os

def causal_holt_smoothing(series, alpha=0.1, beta=0.2):
    """
    Applies a strictly causal Holt linear smoothing filter.
    Returns an array of the same length, containing the smoothed historical states.
    """
    n = len(series)
    smoothed = np.zeros(n)
    
    # Initialization (Month 0)
    level = series[0]
    #Note on fixing early spikes: > 
    # "I noticed a weird artificial spike or drop at the very beginning of the smoothed curve,
    #  it’s because the code is overreacting to the very first data point.
    #  I fixed this by setting the initial trend value to 0 at the first time step. 
    # This stops early noise or zero-count months from throwing off the momentum of the filter."
    trend = 0.0                       #series[1] - series[0] if n > 1 else 0.0 
    smoothed[0] = level
    
    # Sequential filtering across the 20-year timeline
    for t in range(1, n):
        value = series[t]
        last_level = level
        
        # 1. Update the smoothed baseline level for the current month
        level = alpha * value + (1 - alpha) * (level + trend)
        # 2. Update the localized growth trend
        trend = beta * (level - last_level) + (1 - beta) * trend
        
        # 3. Store the clean, current level state (No future leaking)
        smoothed[t] = level
        
    return smoothed

# Configuration
target_files = [
    "scopus_neuromorphic_auxiliary_features.csv"
]
alpha_val = 0.1
beta_val = 0.2

# Execute pipeline across all four target files
for file_name in target_files:
    if not os.path.exists(file_name):
        print(f"Skipping {file_name}: File not found in the working directory.")
        continue
        
    print(f"Processing causal smoothing layer for: {file_name}...")
    df = pd.read_csv(file_name)
    
    # Capture your date/index tracker column safely
    date_col = df.columns[0] 
    
    # Isolate all continuous clinical and public interest features
    numerical_cols = df.columns[1:]
    
    # Map the filter across each feature trajectory independently
    df_smoothed = df.copy()
    for col in numerical_cols:
        # Convert to float to avoid typing issues during calculation
        raw_series = df[col].astype(float).values 
        df_smoothed[col] = causal_holt_smoothing(raw_series, alpha=alpha_val, beta=beta_val)
        
    # Overwrite or output to a distinct file state prior to Global Z-scoring
    output_name = "smoothed_auxiliary.csv"
    df_smoothed.to_csv(output_name, index=False)
    print(f"Successfully saved clean causal timelines to: {output_name}\n")