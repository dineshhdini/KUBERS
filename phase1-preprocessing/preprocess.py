import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load a CSV file into a Pandas DataFrame.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path} with {df.shape[0]} rows.")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
def clean_data(df):
    """
    Clean the DataFrame by handling missing values.
    Args:
        df (pd.DataFrame): Input data.
    Returns:
        pd.DataFrame: Cleaned data.
    """
    # Fill missing numerical values with the column mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Drop rows with remaining NaNs (e.g., in non-numeric columns)
    df.dropna(inplace=True)
    
    print(f"Cleaned data: {df.shape[0]} rows remaining.")
    return df

def normalize_data(df, columns):
    """
    Normalize specified columns to a 0-1 range.
    Args:
        df (pd.DataFrame): Input data.
        columns (list): List of column names to normalize.
    Returns:
        pd.DataFrame: Normalized data.
    """
    for col in columns:
        if col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            print(f"Warning: Column {col} not found in DataFrame.")
    return df

if __name__ == "__main__":
    # Load sample data from CSV
    df = load_data("sample_k8s_data.csv")
    if df is not None:
        # Test the pipeline
        df = clean_data(df)
        df = normalize_data(df, ["cpu_usage", "memory_usage"])
        print("Processed sample data:")
        print(df)