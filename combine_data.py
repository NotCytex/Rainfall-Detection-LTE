import os
import pandas as pd

# Define paths to your folders
rain_folder = 'data/uts-rain'
no_rain_folder = 'data/uts'

def load_and_label_data(folder_path, label):
    """
    Load CSV files from the specified folder, add a label column, and concatenate into a single DataFrame.

    Args:
    folder_path (str): Path to the folder containing the CSV files.
    label (int): Label to add to each row (1 for rain, 0 for no rain).

    Returns:
    pd.DataFrame: Combined DataFrame with data and labels.
    """
    data_frames = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, header=None)  # Adjust header as per your CSV file format
            df['label'] = label
            data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

# Load and label data
rain_data = load_and_label_data(rain_folder, label=1)  # 1 for rain
no_rain_data = load_and_label_data(no_rain_folder, label=0)  # 0 for no rain


# Combine the data
combined_data = pd.concat([rain_data, no_rain_data], ignore_index=True)

# Save the combined data to a new CSV file
combined_data.to_csv('combined_data.csv', index=False)

print("Data combined and saved to 'combined_data.csv'.")
