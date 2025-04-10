import pandas as pd
import numpy as np
import csv

# Load the dataset
df = pd.read_csv("C:/Users/janak/AI/scienceProject2024/DeepLearWing_v2.csv")

# Function to clean and convert coordinates to numeric arrays
def clean_coords(coords_str):
    # Remove unwanted characters (like brackets or extra spaces) from the string
    coords_str = coords_str.replace('[', '').replace(']', '').replace('\'', '').strip()
    # Split the string into a list and convert each to a float
    return np.array(coords_str.split(), dtype=float)

# Function to compute max thickness and max camber for a given airfoil
def compute_max_thickness_camber(x_coords, y_coords):
    x_coords = clean_coords(x_coords)  # Clean and convert x_coords to numpy array
    y_coords = clean_coords(y_coords)  # Clean and convert y_coords to numpy array
    
    # Ensure that both x_coords and y_coords have the same length
    if len(x_coords) != len(y_coords):
        # If lengths don't match, truncate the longer one to the length of the shorter one
        min_len = min(len(x_coords), len(y_coords))
        x_coords = x_coords[:min_len]
        y_coords = y_coords[:min_len]
    
    # Split the y_coords into upper and lower surfaces
    n = len(x_coords) // 2  # Assuming symmetrical airfoils
    upper_y = y_coords[:n]
    lower_y = y_coords[n:]
    
    # Handle odd-length y_coords by adjusting the split point
    if len(upper_y) != len(lower_y):
        # Trim the longer array to make them equal
        min_len = min(len(upper_y), len(lower_y))
        upper_y = upper_y[:min_len]
        lower_y = lower_y[:min_len]
    
    # Calculate thickness and camber
    thickness = np.abs(upper_y - lower_y)  # Compute thickness as the difference between upper and lower surfaces
    camber = (upper_y + lower_y) / 2  # Camber is the average of upper and lower surfaces
    
    # Calculate max values
    max_thickness = np.max(thickness)
    max_camber = np.max(camber)
    
    # Convert coordinates to comma-separated strings for output
    x_coords_str = ','.join(map(str, x_coords))  # Join x_coords as a comma-separated string
    y_coords_str = ','.join(map(str, y_coords))  # Join y_coords as a comma-separated string
    
    return max_thickness, max_camber, x_coords_str, y_coords_str

# Create an empty list to hold the results for each airfoil
max_values = []

# Iterate through the dataset and compute for each unique airfoil
for index, row in df.iterrows():
    name = row["name"]
    
    # Compute the max thickness, max camber, and cleaned coordinate strings for this airfoil
    max_thickness, max_camber, x_coords_str, y_coords_str = compute_max_thickness_camber(row["x_coords"], row["y_coords"])
    
    # Append the result to the list
    max_values.append({
        'name': name,
        'angle': row['angle'],
        'reynolds': row['reynolds'],
        'x_coords': x_coords_str,  # Use the cleaned comma-separated string
        'y_coords': y_coords_str,  # Use the cleaned comma-separated string
        'cd': row['cd'],
        'cl': row['cl'],
        'cm': row['cm'],
        'cl/cd': row['cl/cd'],
        'max_thickness': max_thickness,
        'max_camber': max_camber
    })

# Create a DataFrame for the max values
max_values_df = pd.DataFrame(max_values)

# Save the updated DataFrame to a new CSV file without quotes and with escape character
output_file = "C:/Users/janak/AI/scienceProject2024/testData_max_values.csv"
max_values_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

# Optional: Print the first few rows to verify
print(max_values_df.head())
