import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Load your trained model (ensure the model path is correct)
MODEL_PATH = 'path_to_your_saved_model'
cnn = tf.keras.models.load_model('my_model.h5')

# Odia letters mapping (as defined in your model)
odia_mapping = {
    0: 'ଅ', 1: 'ଆ', 2: 'ଇ', 3: 'ଈ', 4: 'ଉ', 5: 'ଊ', 6: 'ଋ',
    7: 'ଏ', 8: 'ଐ', 9: 'ଓ', 10: 'ଔ', 11: 'କ', 12: 'ଖ', 13: 'ଗ',
    14: 'ଘ', 15: 'ଙ', 16: 'ଚ', 17: 'ଛ', 18: 'ଜ', 19: 'ଝ', 20: 'ଞ',
    21: 'ଟ', 22: 'ଠ', 23: 'ଡ', 24: 'ଢ', 25: 'ଣ', 26: 'ତ', 27: 'ଥ',
    28: 'ଦ', 29: 'ଧ', 30: 'ନ', 31: 'ପ', 32: 'ଫ', 33: 'ବ', 34: 'ଭ',
    35: 'ମ', 36: 'ଯ', 37: 'ର', 38: 'ଲ', 39: 'ଳ', 40: 'ଵ', 41: 'ଶ',
    42: 'ଷ', 43: 'ସ', 44: 'ହ', 45: '଼', 46: 'ା'
}
 # Include your mapping here

st.title('Odia Letter Classifier')

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Upload an image of the Odia letter", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = image.load_img(uploaded_file, target_size=(64, 64))

    # Preprocess the image for the model
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    # Make prediction
    prediction = cnn.predict(img)

    # Convert prediction to readable letter
    predicted_class = np.argmax(prediction)
    predicted_odia_letter = odia_mapping[predicted_class]

    # Display the prediction
    st.write(f"Predicted Odia Letter: {predicted_odia_letter}")
