import pandas as pd

def load_data(file_path):
    """
    Load the combined CSV file and return the features and labels.
    
    Args:
    file_path (str): Path to the CSV file.

    Returns:
    tuple: Features (X) and labels (y) as numpy arrays.
    """
    combined_data = pd.read_csv(file_path)
    X = combined_data.iloc[:, :-1].values  # Features
    y = combined_data.iloc[:, -1].values   # Labels
    return X, y
