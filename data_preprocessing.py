from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def normalize_data(X):
    """
    Normalize the feature data.

    Args:
    X (numpy.ndarray): The feature data.

    Returns:
    numpy.ndarray: Normalized feature data.
    """
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Args:
    X (numpy.ndarray): The feature data.
    y (numpy.ndarray): The labels.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    tuple: Training and testing data and labels.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
