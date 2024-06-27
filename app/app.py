from keras.models import load_model
from PIL import Image
import numpy as np
import streamlit as st
from PIL import Image
import numpy as np
import os 

os.chdir('/Users/aladelca/Library/CloudStorage/OneDrive-McGillUniversity/portfolio/dragon_ball_image_recognition/dragon_ball_image_recognition/')


def load_trained_model(path):
    model = load_model(path)
    return model

def preprocessing(img):
    img = Image.open(img).convert('RGB')  # Convert to RGB to ensure uniformity
    img = img.resize((256,256))
    img = np.array(img)
    return img

def predict_image(model, image):
    prediction = model.predict(image)
    return prediction



characters = ['frieza', 'muten roshi', 'son gohan','son goku','vegeta']
st.title("Model deployment: Dragon Ball image recognition")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    
    
    st.image(image, caption='Image', use_column_width=True)
    img = preprocessing(uploaded_file)
    img = np.expand_dims(img, axis=0)

    model = load_trained_model('model/final_model.h5')
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    st.write(f"Prediction: {characters[predicted_class]}")
    
else:
    st.write("Load an image to continue.")


