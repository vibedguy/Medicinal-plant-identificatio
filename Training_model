#run this
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the paths to the train and test directories
train_dir = '/content/drive/MyDrive/Project/Train'
test_dir = '/content/drive/MyDrive/Project/Test'

# Image size and batch size
image_size = (224, 224)
batch_size = 32

# Create ImageDataGenerators for train and test datasets
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Scale pixel values to [0, 1]
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load and preprocess the train dataset
print("Loading train database...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary"  # Use "categorical" if you have more than two classes
)

# Define a CNN model for training
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Use "softmax" if you have more than two classes
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10  # You can adjust the number of epochs
print("Training the model...")
model.fit(train_generator, epochs=epochs)

# Save the trained model for later use
model.save('plant_classification_model.h5')

print("Training complete. Model saved as 'plant_classification_model.h5")
