import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2

# Load trained model
model = load_model("digit_model.h5")

st.title("✍ Handwritten Digit Digitizer")
st.write("Draw a digit (0-9) in the box below")

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to 28x28
        img = cv2.resize(img, (28, 28))
        
        # Invert colors
        img = cv2.bitwise_not(img)
        
        # Normalize
        img = img / 255.0
        
        # Reshape
        img = img.reshape(1, 28, 28, 1)
        
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        
        st.success(f"Predicted Digit: {predicted_digit}")
