import requests
import streamlit as st
import streamlit_drawable_canvas as draw
import tensorflow as tf
import numpy as np
import cv2



st.title("MNIST Digit Recognizer")
size = 192

canvas_result = draw.st_canvas(
    fill_color = "#ffffff",
    stroke_width = 20,
    stroke_color = "#ffffff",
    background_color = "#000000",
    height = 150,
    width = 150,
    drawing_mode = "freedraw",
    key = "canvas")

if canvas_result.image_data is not None:
    img_color= cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img_rescaling = cv2.resize(img_color, (size, size), interpolation=cv2.INTER_NEAREST)
    st.write('Input Image')
    st.image(img_rescaling)

if st.button('Predict'):
        url = 'http://localhost:8000/predict'
        data_sent = cv2.imencode('.png', img_color)[1].tobytes()
        files = {'img': data_sent}
        response = requests.post(url, files=files)
        data_recieved = response.json()
        result = data_recieved['result']
        percent = data_recieved['percent']
        st.write(f'result: {result}')
        st.bar_chart(percent)


