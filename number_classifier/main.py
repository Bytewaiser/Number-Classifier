import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

st.title("Number Classifier")

# Initialization
if 'model' not in st.session_state:
    st.session_state['model'] = load_model("2_Layer_Conv_15_Epoch")

col_1, col_2 = st.columns(2)
# Specify canvas parameters in application
# Create a canvas component
with col_1:
    canvas_result = st_canvas(
    stroke_width=25,
    stroke_color="rgb(255, 255, 255)",
    background_color="rgb(0, 0, 0)",
    background_image=None,
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    point_display_radius=0,
    key="canvas",
)


a = canvas_result.image_data[:,:,:-1]
a = Image.fromarray(a).resize((28, 28))
a = np.array(a).mean(axis=2).astype(np.uint8)
arr = a.reshape(1, 28, 28, 1) / 255

model = st.session_state["model"]
p = model.predict(arr, verbose=0)
with col_2:
    st.bar_chart(p.T)
st.write("Your number is ", p.argmax())
st.write("Note: Only drawback for this app is drawings is not the same resolution with the traninig data")
st.write("Because of that reason predictions may not be good")
