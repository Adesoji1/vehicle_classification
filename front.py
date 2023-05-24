import tensorflow as tf
from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy as np
import requests
import streamlit as st


# Set the page configuration
st.set_page_config(
    page_title="Vehicle Image Classification",
    
)

# Define the CSS styles
main_style = """
    body {
        background-image: url('https://unsplash.com/photos/Wi26sbU-2F0');
        background-size: cover;
    }
"""

# Apply the CSS styles
st.markdown(f'<style>{main_style}</style>', unsafe_allow_html=True)

# Display the title
st.title("Vehicle Image Classification")

# Upload the image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Make a POST request to the FastAPI backend
    response = requests.post("http://localhost:8000/predict", files={"file": uploaded_file})

    if response.status_code == 200:
        try:
            data = response.json()
            # Display the prediction
            prediction_label = data["predicted_label"]
            prediction_confidence = data["confidence"]

            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.write("Prediction Label:", prediction_label)
            st.write("Confidence:", prediction_confidence)

        except ValueError as e:
            st.error("Error: Failed to decode the response JSON.")
            st.error(f"Response content: {response.content}")

    else:
        st.error(f"Error: Request failed with status code {response.status_code}")
        st.error(f"Response content: {response.content}")
