import streamlit as st
import tensorflow as tf
from PIL import Image

# Define the class labels
class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

# Load the best model
best_model = tf.keras.models.load_model('/content/drive/MyDrive/Colab/Weather/best_model.h5')

# Define the app layout
st.title('Weather Classification')
upload_file = st.file_uploader('Upload an image', type=['png', 'jpg'])

if upload_file is not None:
    # Preprocess the image
    image = Image.open(upload_file)
    image = image.resize((32, 32))  # Resize to the input dimensions of your model
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array.reshape((1,) + image_array.shape) / 255.0  # Normalize pixel values

    # Make a prediction
    prediction = best_model.predict(image_array)
    predicted_class = class_labels[prediction.argmax()]

    # Display the results
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted Weather: {predicted_class}')
