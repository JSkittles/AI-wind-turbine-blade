import pandas as pd
from sklearn.preprocessing import StandardScaler
from ctgan.synthesizers import CTGAN
import joblib  # for saving the model and scaler
import numpy as np

# Load the flattened dataset
data = pd.read_csv("C:/Users/janak/AI/scienceProject2024/CTGAN/flattenedAirfoilData.csv", header=None)

# Extract the features for CTGAN training (excluding the airfoil ID)
data_without_ids = data.iloc[:, 1:]  # All columns except the first (features for training)

# Initialize the scaler (StandardScaler scales data to have zero mean and unit variance)
scaler = StandardScaler()

# Fit and transform the data without the airfoil IDs (features for CTGAN model)
data_scaled = scaler.fit_transform(data_without_ids)

# Define and train the CTGAN with the scaled data
ctgan = CTGAN(epochs=300)  # Set the number of epochs to a desired value
ctgan.fit(data_scaled)

# Save the fitted scaler for future use
joblib.dump(scaler, 'C:/Users/janak/AI/scienceProject2024/CTGAN/scaler.pkl')

# Save the trained CTGAN model for future use
joblib.dump(ctgan, 'C:/Users/janak/AI/scienceProject2024/CTGAN/ctgan_model.pkl')

print("CTGAN model and scaler have been saved.")