import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model_final.h5')
    return model

model = load_model()

# Set wallpaper
st.markdown(
    """
    <style>
    body {
        background: url('https://w0.peakpx.com/wallpaper/344/679/HD-wallpaper-gojousatoru-anime-gojou-satoru-jujutsu-kaisen.jpg') no-repeat center center fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    
# Clear Button
if st.button("Clear"):
    file = None
    st.text("Please upload an image file")
    
# Adding a sidebar
st.sidebar.title("Menu")
selected_option = st.sidebar.selectbox("Select", ("About", "Help", "Visualization", "Conclusion"))

if selected_option == "About":
    st.sidebar.write("This Final application was created by:")
    st.sidebar.write("- Name: Bencito, Sonny Jay")
    st.sidebar.write("- Section and Grade: CPE32S4")
    st.sidebar.write("- Instructor: Dr. Jonathan Taylar")

elif selected_option == "Help":
    st.sidebar.write("Upload an image from the Fashion MNIST dataset and click 'Predict' to see the predicted category.")

elif selected_option == "Visualization":
    st.sidebar.write("Visualize the uploaded image")
    if file is not None:
        st.sidebar.subheader("Uploaded Image")
        st.sidebar.image(image)

elif selected_option == "Conclusion":
    st.sidebar.write("Thank you for using this application. We hope it has been useful to you.")
