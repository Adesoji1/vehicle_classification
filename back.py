import tensorflow as tf
from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy as np
import requests


# Determine the device to run the model on
device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"

# Load the model
model_path = "brandNet.model"
model = tf.keras.models.load_model(model_path, compile=False)

# Define the FastAPI app
app = FastAPI()

# Define the predict endpoint
@app.post("/predict")
async def predict_image(file: UploadFile):
    # Convert the file to PIL Image
    image = Image.open(file.file).convert("RGB")

    # Preprocess the image
    target_width = 180
    target_height = 180
    image = image.resize((target_width, target_height))
    image_array = np.array(image) / 255.0
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    # Run the model prediction
    with tf.device(device):
        predictions = model.predict(image_tensor)
        scores = tf.nn.softmax(predictions[0])

    # Load class names (replace with your own class names if needed)
    class_names = ['Audi', 'BMW', 'Chevrolet', 'acura', 'bugatti', 'chery', 'chrysler', 'citroen', 'daewoo', 'daihatsu']

    # Get the predicted label and confidence
    predicted_label = class_names[np.argmax(scores)]
    confidence = np.max(scores) * 100

    # Format the confidence as a percentage
    confidence_percent = "{:.2f}%".format(confidence)

    # Return the prediction result
    return {"predicted_label": predicted_label, "confidence": confidence_percent}