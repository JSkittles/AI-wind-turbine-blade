import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.signal import savgol_filter
import tensorflow as tf

# Load CTGAN model and scaler from the provided file paths
ctgan = joblib.load('C:/Users/janak/AI/scienceProject2024/CTGAN/ctgan_model.pkl')
scaler = joblib.load('C:/Users/janak/AI/scienceProject2024/CTGAN/scaler.pkl')

# Load RNN Models
model = tf.keras.models.load_model('C:/Users/janak/AI/scienceProject2024/RNN/airfoil_rnn_model.h5')
scalerRNN = joblib.load('C:/Users/janak/AI/scienceProject2024/RNN/scaler.pkl')

# Parameters
num_airfoils = 1
num_points = 1000 # High resolution for smoothness
i = 1

maxClCd = 0
max_x_coords = []
max_y_upper = []
max_y_lower = []

while i <= num_airfoils:
    print(f"Airfoil Number: {i}\n")

    # Generate synthetic data using CTGAN
    synthetic_data_scaled = ctgan.sample(num_points)
    synthetic_data = scaler.inverse_transform(synthetic_data_scaled)

    # Extract y_upper and y_lower from the synthetic data
    y_upper_ctgan = np.clip(synthetic_data[:, 1], 0.0, 1.0)  # Upper surface: Positive
    y_lower_ctgan = np.clip(synthetic_data[:, 2], -1.0, 0.0)  # Lower surface: Negative

    # Ensure the length of x_coords matches the number of generated points
    num_points = min(len(y_upper_ctgan), len(y_lower_ctgan))
    x_coords = np.linspace(0, 1, num_points).reshape(-1, 1)

    # Smooth the data using a Savitzky-Golay filter
    window_size = 1000  # Window size must be odd
    poly_order = 2
    y_upper_smooth = savgol_filter(y_upper_ctgan, window_size, poly_order)
    y_lower_smooth = savgol_filter(y_lower_ctgan, window_size, poly_order)

    # Normalize and center the y-coordinates
    y_upper_smooth -= y_upper_smooth[0]  # Ensure the upper surface starts from 0
    y_upper_smooth /= 2  # Scale the upper surface

    y_lower_smooth -= y_lower_smooth[0]  # Ensure the lower surface starts from 0
    y_lower_smooth /= 1.5  # Scale the lower surface

    # Ensure that upper surface is positive and lower surface is negative
    y_upper_smooth = np.abs(y_upper_smooth)  # Force upper surface to be positive
    y_lower_smooth = -np.abs(y_lower_smooth)  # Force lower surface to be negative

    # Ensure the upper and lower surfaces meet at the trailing edge
    def taper_to_zero(y, taper_fraction=0.5):
        taper_length = int(len(y) * taper_fraction)
        taper_weights = np.linspace(1, 0, taper_length)
        y[-taper_length:] *= taper_weights
        return y

    y_upper_smooth = taper_to_zero(y_upper_smooth)
    y_lower_smooth = taper_to_zero(y_lower_smooth)

    # Function to remove jagged points
    def remove_jagged_points(x, y):
        dy = np.diff(y)
        d2y = np.diff(dy)
        smooth_indices = [0]
        for j in range(1, len(d2y)):
            if np.sign(d2y[j - 1]) != np.sign(d2y[j]):
                continue
            smooth_indices.append(j + 1)
        return x[smooth_indices], y[smooth_indices]

    x_filtered, y_upper_filtered = remove_jagged_points(x_coords.flatten(), y_upper_smooth)
    x_filtered, y_lower_filtered = remove_jagged_points(x_coords.flatten(), y_lower_smooth)

    min_len = min(len(x_filtered), len(y_upper_filtered), len(y_lower_filtered))
    x_filtered, y_upper_filtered, y_lower_filtered = (
        x_filtered[:min_len],
        y_upper_filtered[:min_len],
        y_lower_filtered[:min_len],
    )

    # Function to calculate camber and thickness
    def calculate_camber_thickness(x, y_upper, y_lower):
        camber = [(yu + yl) / 2 for yu, yl in zip(y_upper, y_lower)]
        thickness = [abs(yu - yl) for yu, yl in zip(y_upper, y_lower)]
        return max(camber), max(thickness)

    max_camber, max_thickness = calculate_camber_thickness(x_filtered, y_upper_filtered, y_lower_filtered)
    print(f"Maximum Camber: {max_camber}")
    print(f"Maximum Thickness: {max_thickness}")

    # Input for RNN model
    y_coords_combined = np.concatenate((y_upper_filtered, y_lower_filtered))

    aoa = 18
    reynolds = 1.5e6

    new_features = [aoa, reynolds, max_thickness, max_camber]
    features_scaled = scalerRNN.transform([new_features])
    x_reshaped = np.array(x_filtered).reshape(1, len(x_filtered), 1)
    y_reshaped = np.array(y_coords_combined).reshape(1, len(y_coords_combined), 1)

    predictions = model.predict({'input_1': x_reshaped, 'input_2': y_reshaped})
    cd, cl, cm = predictions[0]
    cl_cd_ratio = cl / cd if cd != 0 else float('inf')
    print(f"Cl/Cd Ratio: {cl_cd_ratio}")

    if cl_cd_ratio > maxClCd:
        maxClCd = cl_cd_ratio
        max_x_coords = x_filtered
        max_y_upper = y_upper_filtered
        max_y_lower = y_lower_filtered

    i += 1

print("\nMax Cl/Cd Ratio: ", maxClCd)
print("\nMax Y Upper Surface: ", ', '.join(map(str, max_y_upper)))
print("\n\n\n\n\n\n\n\n\nMax Y Lower Surface: ", ', '.join(map(str, max_y_lower)))


plt.figure(figsize=(10, 5))
plt.plot(max_x_coords, max_y_upper, label="Upper Surface", color="blue")
plt.plot(max_x_coords, max_y_lower, label="Lower Surface", color="red")
plt.scatter(1.0, 0.0055, color='red', s=100, label="Highlight")  
plt.scatter(1.0, -0.003, color='red', s=100, label="Highlight")  

plt.title("Optimized Airfoil Shape")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
plt.legend()
plt.grid() 
plt.show()
