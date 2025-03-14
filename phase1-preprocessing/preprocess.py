import pandas as pd
import numpy as np

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path} with {df.shape[0]} rows.")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
def clean_data(df):
    # Fill missing numerical values with the column mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Drop rows with remaining NaNs (e.g., in non-numeric columns)
    df.dropna(inplace=True)
    
    print(f"Cleaned data: {df.shape[0]} rows remaining.")
    return df

def normalize_data(df, columns):
    
    for col in columns:
        if col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            print(f"Warning: Column {col} not found in DataFrame.")
    return df

def extract_features(df, window=5):
    """
    Extract time-series features from the DataFrame.
    Args:
        df (pd.DataFrame): Normalized data.
        window (int): Rolling window size for averages.
    Returns:
        pd.DataFrame: Data with added features.
    """
    # Ensure timestamp is in datetime format for time-series operations
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Rolling average for numerical columns
    numeric_cols = ["cpu_usage", "memory_usage"]
    for col in numeric_cols:
        if col in df.columns:
            df[f"{col}_rolling_avg"] = df[col].rolling(window=window, min_periods=1).mean()
    
    # Spike detection (difference exceeds threshold)
    for col in numeric_cols:
        if col in df.columns:
            df[f"{col}_spike"] = df[col].diff().abs() > 0.2  # Threshold: 20% change
    
    # Optional: Count 'Failed' statuses (if pod_status exists)
    if "pod_status" in df.columns:
        df["failure_flag"] = df["pod_status"].apply(lambda x: 1 if x == "Failed" else 0)
    
    print(f"Extracted features for {len(df)} rows.")
    return df

if __name__ == "__main__":
    # Sample Kubernetes-like data
    sample_data = pd.DataFrame({
        "timestamp": [
            "2025-03-14 10:00:00",
            "2025-03-14 10:01:00",
            "2025-03-14 10:02:00",
            "2025-03-14 10:03:00",
            "2025-03-14 10:04:00"
        ],
        "cpu_usage": [50.0, 70.0, np.nan, 90.0, 60.0],
        "memory_usage": [200.0, 250.0, 300.0, 280.0, 220.0],
        "pod_status": ["Running", "Running", "Failed", "Running", "Running"]
    })
    
    # Test the pipeline
    df = clean_data(sample_data)
    df = normalize_data(df, ["cpu_usage", "memory_usage"])
    df = extract_features(df, window=3)
    print("Processed sample data with features:")
    print(df)