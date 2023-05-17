import tensorflow as tf
import numpy as np
import streamlit as st

# Load the best model
best_model = tf.keras.models.load_model('/content/drive/MyDrive/Colab/mnist/best_model_final.h5')

# Streamlit code to showcase the best model
st.title("Fashion MNIST Classifier")
image_file = st.file_uploader("/content/drive/MyDrive/Colab/mnist/tshirt1.jpeg", type=["png", "jpg"])

if image_file is not None:
    # Read the uploaded image
    image = tf.keras.preprocessing.image.load_img(image_file, target_size=(28, 28), color_mode='grayscale')
    input_array = tf.keras.preprocessing.image.img_to_array(image)
    input_array = input_array / 255.0
    input_array = np.expand_dims(input_array, 0)
    
    # Classify the image using the best model
    prediction = best_model.predict(input_array)
    predicted_class = np.argmax(prediction)
    
    # Display the predicted class
    st.image(image, caption=f"Predicted Class: {predicted_class}", width=200)

