import numpy as np

def process_airfoil_row(row, airfoil_id):
    # Find the indices of the two occurrences of 1.0 in the row (for x-coordinates)
    first_one_index = row.index(1.0)  # First occurrence of 1.0
    second_one_index = row.index(1.0, first_one_index + 1)  # Second occurrence of 1.0

    # Extract x-coordinates from first 1.0 to second 1.0
    x_coords = row[first_one_index:second_one_index]
    
    # Extract y-coordinates starting from the first 0.0 up to the end of the row
    y_coords = row[second_one_index+1:]  # All coordinates after the first 1.0 are y-coordinates
    
    # Split y-coordinates into upper and lower surfaces
    midpoint = len(y_coords) // 2
    y_upper = y_coords[:midpoint]  # First half for upper surface
    y_lower = y_coords[midpoint:]  # Second half for lower surface
    
    flattened_x = []
    flattened_y_upper = []
    flattened_y_lower = []
    ids = []

    # Loop through x-coordinates and their corresponding y-upper and y-lower
    for x, y_u, y_l in zip(x_coords, y_upper, y_lower):
        flattened_x.append(x)  # Add the x-coordinate once
        flattened_y_upper.append(y_u)  # Add the corresponding y-coordinate for upper surface
        flattened_y_lower.append(y_l)  # Add the corresponding y-coordinate for lower surface
        ids.append(airfoil_id)  # Add the airfoil ID for each coordinate pair

    return flattened_x, flattened_y_upper, flattened_y_lower, ids

def read_input_csv(file_path):
    with open(file_path, 'r') as file:
        # Read the file and split by lines
        lines = file.readlines()
    
    all_x = []
    all_y_upper = []
    all_y_lower = []
    all_ids = []

    airfoil_id = 1  # Start with airfoil ID 1, and increment for each new airfoil
    
    for line in lines:
        # Skip lines that do not contain numeric values (headers or malformed lines)
        try:
            # Split the line by commas and convert the values to floats
            row = list(map(float, line.strip().split(',')))
        except ValueError:
            # If a ValueError occurs, skip this line (e.g., it could be a header or non-numeric row)
            print(f"Skipping invalid row: {line.strip()}")
            continue
        
        # Process the airfoil row to get the flattened data with id
        x, y_upper, y_lower, ids = process_airfoil_row(row, airfoil_id)
        
        all_x.extend(x)
        all_y_upper.extend(y_upper)
        all_y_lower.extend(y_lower)
        all_ids.extend(ids)
        
        airfoil_id += 1  # Increment airfoil ID for the next airfoil
    
    return all_x, all_y_upper, all_y_lower, all_ids


def save_flattened_data(all_x, all_y_upper, all_y_lower, all_ids, output_file):
    # Combine the airfoil ID, x-coordinates, y-upper, and y-lower into one array
    flattened_data = np.column_stack((all_ids, all_x, all_y_upper, all_y_lower))
    
    # Save to CSV without headers
    with open(output_file, 'w') as file:
        for data in flattened_data:
            file.write(','.join(map(str, data)) + '\n')

# File paths (modify as necessary)
input_file = 'C:/Users/janak/AI/scienceProject2024/CTGAN/coordinatesForCTGAN.csv'  # Path to the input CSV file
output_file = 'C:/Users/janak/AI/scienceProject2024/CTGAN/flattenedAirfoilData.csv'  # Path to the output CSV file

# Read input, process, and save output
all_x, all_y_upper, all_y_lower, all_ids = read_input_csv(input_file)
save_flattened_data(all_x, all_y_upper, all_y_lower, all_ids, output_file)

print("Flattened airfoil data with airfoil ID, x-coordinates, y-upper, and y-lower has been saved to", output_file)
