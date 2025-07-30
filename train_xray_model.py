import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Path to dataset
base_dir = 'chest_xray'

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Image preprocessing
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(150, 150), class_mode='binary', batch_size=32)
val_data = val_gen.flow_from_directory(val_dir, target_size=(150, 150), class_mode='binary', batch_size=32)
test_data = test_gen.flow_from_directory(test_dir, target_size=(150, 150), class_mode='binary', batch_size=32)

# Model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary output: 0 (Normal) or 1 (Pneumonia)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_data, validation_data=val_data, epochs=5)

# Save
model.save("xray_model.h5")
print("✅ Model saved as xray_model.h5")