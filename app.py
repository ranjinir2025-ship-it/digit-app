import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = load_model("digit_model.h5")

st.set_page_config(page_title="Handwritten Digit Digitizer")

st.title("✍️ Handwritten Digit Digitizer")
st.write("Draw a digit (0–9) in the box below")

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28))

        # Invert colors (MNIST style)
        inverted = cv2.bitwise_not(resized)

        # Normalize
        normalized = inverted / 255.0

        # Reshape for model
        reshaped = normalized.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(reshaped)
        digit = np.argmax(prediction)

        st.success(f"Predicted Digit: {digit}")

