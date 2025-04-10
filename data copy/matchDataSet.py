import pandas as pd
import numpy as np

# Load the dataset
file_path = "C:/Users/janak/AI/scienceProject2024/DeepLearWing_v2.csv"
data = pd.read_csv(file_path)

# Parse x_coords and y_coords into arrays
data['x_coords'] = data['x_coords'].apply(lambda x: np.fromstring(x, sep=','))
data['y_coords'] = data['y_coords'].apply(lambda y: np.fromstring(y, sep=','))

# Filter rows where the lengths of x_coords and y_coords are equal
filtered_data = data[data['x_coords'].apply(len) == data['y_coords'].apply(len)]

# Save the filtered dataset to a new CSV (optional)
filtered_file_path = "C:/Users/janak/AI/scienceProject2024/Filtered_DeepLearWing.csv"
filtered_data.to_csv(filtered_file_path, index=False)

# Print the number of filtered rows
print(f"Filtered dataset contains {len(filtered_data)} rows out of {len(data)}.")
