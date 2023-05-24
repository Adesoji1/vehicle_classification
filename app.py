# import tensorflow as tf
# from fastapi import FastAPI, UploadFile
# from PIL import Image
# import numpy as np
# import requests
# import streamlit as st

# # Determine the device to run the model on
# device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"

# # Load the model
# model_path = "brandNet.model"
# model = tf.keras.models.load_model(model_path, compile=False)

# # Define the FastAPI app
# app = FastAPI()

# # Define the predict endpoint
# @app.post("/predict")
# async def predict_image(file: UploadFile):
#     # Convert the file to PIL Image
#     image = Image.open(file.file).convert("RGB")

#     # Preprocess the image
#     target_width = 180
#     target_height = 180
#     image = image.resize((target_width, target_height))
#     image_array = np.array(image) / 255.0
#     image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
#     image_tensor = tf.expand_dims(image_tensor, axis=0)

#     # Run the model prediction
#     with tf.device(device):
#         predictions = model.predict(image_tensor)
#         scores = tf.nn.softmax(predictions[0])

#     # Load class names (replace with your own class names if needed)
#     class_names = ['Audi', 'BMW', 'Chevrolet', 'acura', 'bugatti', 'chery', 'chrysler', 'citroen', 'daewoo', 'daihatsu']

#     # Get the predicted label and confidence
#     predicted_label = class_names[np.argmax(scores)]
#     confidence = np.max(scores) * 100

#     # Format the confidence as a percentage
#     confidence_percent = "{:.2f}%".format(confidence)

#     # Return the prediction result
#     return {"predicted_label": predicted_label, "confidence": confidence_percent}


# # Set the page configuration
# st.set_page_config(
#     page_title="Vehicle Image Classification",
#     layout="wide",
    
# )

# # Define the CSS styles
# main_style = """
#     body {
#         background-image: url('https://unsplash.com/photos/Wi26sbU-2F0');
#         background-size: cover;
#     }
# """

# # Apply the CSS styles
# st.markdown(f'<style>{main_style}</style>', unsafe_allow_html=True)

# # Display the title
# st.title("Vehicle Image Classification")

# # Upload the image
# uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Make a POST request to the FastAPI backend
#     response = requests.post("http://localhost:8000/predict", files={"file": uploaded_file})

#     if response.status_code == 200:
#         try:
#             data = response.json()
#             # Display the prediction
#             prediction_label = data["predicted_label"]
#             prediction_confidence = data["confidence"]

#             st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
#             st.write("Prediction Label:", prediction_label)
#             st.write("Confidence:", prediction_confidence)

#         except ValueError as e:
#             st.error("Error: Failed to decode the response JSON.")
#             st.error(f"Response content: {response.content}")

#     else:
#         st.error(f"Error: Request failed with status code {response.status_code}")
#         st.error(f"Response content: {response.content}")
        
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import requests
# import streamlit as st

# # Determine the device to run the model on
# device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"

# # Load the model
# model_path = "brandNet.model"
# model = tf.keras.models.load_model(model_path, compile=False)

# # Set the page configuration
# st.set_page_config(
#     page_title="Vehicle Image Classification",
#     layout="wide"
# )

# # Define the CSS styles
# main_style = """
#     body {
#         background-image: url('https://unsplash.com/photos/Wi26sbU-2F0');
#         background-size: cover;
#     }
# """

# # Apply the CSS styles
# st.markdown(f'<style>{main_style}</style>', unsafe_allow_html=True)

# # Display the title
# st.title("Vehicle Image Classification")

# # Upload the image
# uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert the file to PIL Image
#     image = Image.open(uploaded_file).convert("RGB")

#     # Preprocess the image
#     target_width = 180
#     target_height = 180
#     image = image.resize((target_width, target_height))
#     image_array = np.array(image) / 255.0
#     image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
#     image_tensor = tf.expand_dims(image_tensor, axis=0)

#     # Run the model prediction
#     with tf.device(device):
#         predictions = model.predict(image_tensor)
#         scores = tf.nn.softmax(predictions[0])

#     # Load class names (replace with your own class names if needed)
#     class_names = ['Audi', 'BMW', 'Chevrolet', 'acura', 'bugatti', 'chery', 'chrysler', 'citroen', 'daewoo', 'daihatsu']

#     # Get the predicted label and confidence
#     predicted_label = class_names[np.argmax(scores)]
#     confidence = np.max(scores) * 100

#     # Format the confidence as a percentage
#     confidence_percent = "{:.2f}%".format(confidence)

#     # Display the prediction
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
#     st.write("Prediction Label:", predicted_label)
#     st.write("Confidence:", confidence_percent)

import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import streamlit as st

def add_bg_from_url():
    return "https://cdn.pixabay.com/photo/2018/05/18/12/24/rain-3411068_960_720.jpg"

background_image_url = add_bg_from_url()

# Determine the device to run the model on
device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"

# Load the model
model_path = "brandNet.model"
model = tf.keras.models.load_model(model_path, compile=False)

# Set the page configuration
st.set_page_config(
    page_title="Vehicle Image Classification",
    
)

# Define the CSS styles
main_style = f"""
    <style>
    .stApp {{
        background-image: url('{background_image_url}');
        background-attachment: fixed;
        background-size: cover;
    }}

    .predicted-label {{
        color: #FFC300;
        font-weight: bold;
    }}

    .confidence {{
        font-weight: bold;
    }}

    .confidence.red {{
        color: red;
    }}

    .confidence.green {{
        color: green;
    }}
    </style>
"""

# Apply the CSS styles
st.markdown(main_style, unsafe_allow_html=True)

# Display the title
st.title("Vehicle Image Classification")

# Upload the image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to PIL Image
    image = Image.open(uploaded_file).convert("RGB")

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

    # Determine confidence color
    confidence_color = "red" if confidence < 50 else "green"

    # Display the prediction
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Prediction Label:", f'<span class="predicted-label">{predicted_label}</span>', unsafe_allow_html=True)
    st.write("Confidence:", f'<span class="confidence {confidence_color}">{confidence_percent}</span>', unsafe_allow_html=True)
