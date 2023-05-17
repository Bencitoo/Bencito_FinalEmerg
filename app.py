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

def import_and_predict(image_data, model):
    # Preprocess the image
    image = image_data.convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    prediction = model.predict(image)
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    predicted_class = np.argmax(prediction)
    output = f"Prediction: {class_names[predicted_class]}"
    return output

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    output = import_and_predict(image, model)
    st.success(output)

