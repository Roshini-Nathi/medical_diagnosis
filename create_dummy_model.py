# create_dummy_model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
import numpy as np

print("Creating a dummy xray_model.h5...")

# Define input shape expected by MobileNetV2
input_shape = (224, 224, 3)

# Load MobileNetV2 base model (excluding the top classification layer)
# We use include_top=False to get the convolutional base features
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the base model layers so they are not retrained
base_model.trainable = False

# Add custom classification head on top of the base model
x = base_model.output
# This is the crucial part: GlobalAveragePooling2D flattens the 3D feature map
# into a 1D vector suitable for dense layers.
x = GlobalAveragePooling2D()(x)
# Add a dense layer for further processing (optional, but common)
x = Dense(128, activation='relu')(x)
# Output layer for binary classification (Pneumonia/Normal)
output = Dense(1, activation='sigmoid')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
# For a real model, you would use appropriate optimizer, loss, and metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the model
model.save("xray_model.h5")

print("Dummy xray_model.h5 created successfully!")
print("Model Summary:")
model.summary()