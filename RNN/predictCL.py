import tensorflow as tf
import joblib
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('C:/Users/janak/AI/scienceProject2024/RNN/airfoil_rnn_model.h5')

# Load the scaler
scaler = joblib.load('C:/Users/janak/AI/scienceProject2024/RNN/scaler.pkl')

# Function to get user input for features and coordinates
def get_input_data():
    # Get the feature inputs
    angle = 6
    reynolds = 200000
    max_thickness = 0.0785
    max_camber = 0.0699

    # x_coords as a tuple (no need to split)
    x_coords_input = (1.0,0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.25,0.2,0.15,0.1,0.075,0.05,0.025,0.0125,0.0,0.0125,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0)
    x_coords = [float(x) for x in x_coords_input]  # Already a tuple, so no need to split

    # y_coords as a tuple (no need to split)
    y_coords_input = (0.0016,0.0124,0.0229,0.0428,0.061,0.0771,0.0905,0.1002,0.1048,0.1044,0.1013,0.0934,0.078,0.0664,0.0513,0.0317,0.0193,0.0,-0.005,-0.0042,-0.001,0.0028,0.0068,0.0145,0.0217,0.0282,0.0333,0.0385,0.0386,0.035,0.0286,0.0202,0.01,0.0044,-0.0016)
    y_coords = [float(y) for y in y_coords_input] 

    # Return all inputs as a tuple
    return [angle, reynolds, max_thickness, max_camber], x_coords, y_coords

# Get user input
new_features, new_x_coords, new_y_coords = get_input_data()

# Normalize the features using the saved scaler
features_scaled = scaler.transform([new_features])  # Normalize features

# Reshape to 3D tensors for LSTM (batch_size, timesteps, features)
new_x_reshaped = np.array(new_x_coords).reshape(1, len(new_x_coords), 1)  # (1, timesteps, 1)
new_y_reshaped = np.array(new_y_coords).reshape(1, len(new_y_coords), 1)  # (1, timesteps, 1)

# Convert to TensorFlow tensors
new_x_tensor = tf.convert_to_tensor(new_x_reshaped, dtype=tf.float32)
new_y_tensor = tf.convert_to_tensor(new_y_reshaped, dtype=tf.float32)

# Make predictions using the model
predictions = model.predict(
    {'input_1': new_x_tensor, 'input_2': new_y_tensor}
)

# Output the predictions
print("\nPredictions (cd, cl, cm):")
print(predictions)
