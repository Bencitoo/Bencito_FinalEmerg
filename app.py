import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('best_model_final.h5')
    return model

model = load_model()

st.write("# MNIST Checker by Bencito")

file = st.file_uploader("Choose an image from the Fashion MNIST dataset", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    output = import_and_predict(image, model)
    st.success(output)

# Adding buttons
if st.button('Predict Another Image'):
    st.text("")
    file = st.file_uploader("Choose an image from the Fashion MNIST dataset", type=["jpg", "png"])
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        output = import_and_predict(image, model)
        st.success(output)

# Adding a sidebar
st.sidebar.title("Options")
selected_option = st.sidebar.selectbox("Select an option", ("About", "Help"))

if selected_option == "About":
    st.sidebar.write("This app was created by Bencito, Sonny Jay CPE32S4.")

elif selected_option == "Help":
    st.sidebar.write("Upload an image from the Fashion MNIST dataset and click 'Predict' to see the predicted category.")

