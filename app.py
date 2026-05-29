import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf
model_new = tf.keras.models.load_model('mnist.h5')


st.title("MNIST Digit Recognizer")
size = 192

canvas_result = st_canvas(
    fill_color = "#ffffff",
    stroke_width = 20,
    stroke_color = "#ffffff",
    background_color = "#000000",
    height = 150,
    width = 150,
    drawing_mode = "freedraw",
    key = "canvas")

if canvas_result.image_data is not None:
    img_color = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img_rescaling = cv2.resize(img_color, (size, size), interpolation=cv2.INTER_NEAREST)
    st.write('Input Image')
    st.image(img_rescaling)

if st.button('Predict'):
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_gray = model_new.predict(img_gray.reshape(1, 28, 28))
    st.write(f'result: {np.argmax(img_gray)}')
    st.bar_chart(img_gray[0])
