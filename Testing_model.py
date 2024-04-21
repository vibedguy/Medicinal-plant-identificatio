import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from google.colab import files  # Import files module for Colab

# Load the trained model
model = load_model('plant_classification_model.h5')

# Function to classify a single image
def classify_single_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0  # Normalize pixel values to [0, 1]

    # Make predictions
    predictions = model.predict(x)

    # Interpret the prediction (binary classification)
    if predictions[0] > 0.5:
        return "medicinal"
    else:
        return "non_medicinal"

# Prompt the user to upload an image
print("Upload an image for classification:")
uploaded = files.upload()

# Perform classification on the uploaded image(s)
for filename, content in uploaded.items():
    # Save the uploaded image to a temporary file
    with open(filename, 'wb') as f:
        f.write(content)

    # Classify the uploaded image
    class_label = classify_single_image(filename)
    print(f"Uploaded Image: {filename}, Classification: {class_label}")
    os.remove(filename)
