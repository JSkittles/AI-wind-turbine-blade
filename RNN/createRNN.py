import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
import csv
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Step 1: Read the CSV file
file_path = 'C:/Users/janak/AI/scienceProject2024/data/simpleData.csv'

# Initialize empty lists for features, targets, x_coords, and y_coords
features = []
targets = []
x_coords = []
y_coords = []

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)

    for row in reader:
        feature_row = [float(row[1]), float(row[2]), float(row[-2]), float(row[-1])]
        features.append(feature_row)

        target_row = [float(row[-6]), float(row[-5]), float(row[-4])]

        try:
            x_start_index = row.index('1.0')
            x_end_index = row.index('1.0', x_start_index + 1)

            y_start_index = x_end_index + 1
            y_end_index = len(row) - 6

            x_coord_values = [float(coord) for coord in row[x_start_index:x_end_index + 1]]
            y_coord_values = [float(coord) for coord in row[y_start_index:y_end_index]]

            if 1.0 not in x_coord_values:
                continue

            x_coords.append(x_coord_values)
            y_coords.append(y_coord_values)
            targets.append(target_row)
        except ValueError:
            continue

# Step 2: Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
joblib.dump(scaler, 'C:/Users/janak/AI/scienceProject2024/RNN/scaler.pkl')

# Step 3: Convert coordinates to Ragged Tensors
x_ragged = tf.ragged.constant(x_coords, dtype=tf.float32)
y_ragged = tf.ragged.constant(y_coords, dtype=tf.float32)

x_padded = x_ragged.to_tensor(default_value=0.0)
y_padded = y_ragged.to_tensor(default_value=0.0)

# Step 4: Prepare the dataset
dataset = tf.data.Dataset.from_tensor_slices(({'input_1': x_padded, 'input_2': y_padded}, np.array(targets)))
train_size = int(0.8 * len(targets))
train_dataset = dataset.take(train_size).batch(32)
test_dataset = dataset.skip(train_size).batch(32)

# Step 5: Build the Model
input_x = tf.keras.layers.Input(shape=(None, 1), dtype=tf.float32)
input_y = tf.keras.layers.Input(shape=(None, 1), dtype=tf.float32)

# Add LSTM layers with Dropout
x_lstm = tf.keras.layers.LSTM(128, return_sequences=True)(input_x)
x_lstm = tf.keras.layers.Dropout(0.2)(x_lstm)  # Dropout after first LSTM layer
x_lstm = tf.keras.layers.LSTM(64)(x_lstm)
x_lstm = tf.keras.layers.Dropout(0.2)(x_lstm)

y_lstm = tf.keras.layers.LSTM(128, return_sequences=True)(input_y)
y_lstm = tf.keras.layers.Dropout(0.2)(y_lstm)
y_lstm = tf.keras.layers.LSTM(64)(y_lstm)
y_lstm = tf.keras.layers.Dropout(0.2)(y_lstm)

# Merge and pass through Dense layers
merged = tf.keras.layers.concatenate([x_lstm, y_lstm])
dense = tf.keras.layers.Dense(128, activation='relu')(merged)
dense = tf.keras.layers.Dropout(0.2)(dense)
dense = tf.keras.layers.Dense(64, activation='relu')(dense)
dense = tf.keras.layers.Dense(32, activation='relu')(dense)

output = tf.keras.layers.Dense(3)(dense)  # Final output layer for cd, cl, cm

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model = tf.keras.Model(inputs=[input_x, input_y], outputs=output)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Callbacks: EarlyStopping and ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)

# Step 6: Train the Model
model.fit(train_dataset, validation_data=test_dataset, epochs=50, callbacks=[early_stopping, reduce_lr])

# Step 7: Save the model
model.save('C:/Users/janak/AI/scienceProject2024/RNN/airfoil_rnn_model_v2.h5')

# Evaluate the model
test_loss, test_mae = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")
