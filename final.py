import tensorflow as tf
import numpy as np
import streamlit as st

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define a callback to save the best model based on validation accuracy
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    '/content/drive/MyDrive/Colab/mnist/model_final.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=30, batch_size=128, callbacks=[model_checkpoint])

# Load the best model
best_model = tf.keras.models.load_model('/content/drive/MyDrive/Colab/mnist/model_final.h5')

# Streamlit code to showcase the best model
st.title("Fashion MNIST Classifier")
image_file = st.file_uploader("/content/drive/MyDrive/Colab/mnist/tshirt1.jpeg", type=["png", "jpg"])

if image_file is not None:
    # Read the uploaded image
    image = tf.keras.preprocessing.image.load_img(image_file, target_size=(28, 28), color_mode='grayscale')
    input_array = tf.keras.preprocessing.image.img_to_array(image)
    input_array = tf.expand_dims(input_array, 0) / 255.0
    
    # Classify the image using the best model
    prediction = best_model.predict(input_array)
    predicted_class = np.argmax(prediction)
    
    # Display the predicted class
    st.image(image, caption=f"Predicted Class: {predicted_class}", width=200)
