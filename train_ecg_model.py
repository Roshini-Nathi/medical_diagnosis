import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense

# Load CSV data
df = pd.read_csv('sample_ecg_data.csv')  # Make sure this file exists

# Extract signal and labels
X = np.array(df['signal'])
y = np.array(df['label'])

# Trim both to a multiple of 10
sequence_length = 10
trim_len = len(X) - (len(X) % sequence_length)
X = X[:trim_len]
y = y[:trim_len]

# Reshape X into sequences of 10 time steps (samples, 10, 1)
X = X.reshape(-1, sequence_length, 1)     # e.g., (n_samples, 10, 1)

# For y, take the last label in each 10-sample sequence
y = y.reshape(-1, sequence_length)
y = y[:, -1]   # Get the label for each sequence

# Build CNN model
model = Sequential([
    Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(sequence_length, 1)),
    Flatten(),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=10, verbose=1)
# Save the model after training
model.save("ecg_cnn_model.h5")