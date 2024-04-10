# Imporiting Necessary Libraries
import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import gdown 
from utils import clean_image, get_prediction, make_results

# Define the Google Drive URL of the model.h5 file
google_drive_url = "https://drive.google.com/file/d/1bRYiAO1raPYfByZSi1Ai68VJ5D351oJD/view?usp=sharing"

# Define the local file path where you want to save the downloaded model.h5 file
local_model_path = "model.h5"

# Download the model.h5 file from Google Drive
gdown.download(google_drive_url, local_model_path, quiet=False)

# Loading the Model and saving to cache
@st.cache(allow_output_mutation=True)
def load_model(path):
    
    # Xception Model
    xception_model = tf.keras.models.Sequential([
    tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4,activation='softmax')
    ])


    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet',input_shape=(512, 512, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4,activation='softmax')
    ])

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))

    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)

    outputs = tf.keras.layers.average([densenet_output, xception_output])


    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Loading the Weights of Model
    model.load_weights(path)
    
    return model


# Removing Menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Loading the Model
model = load_model(local_model_path)

# Title and Description
st.title('Created by Department of Computer Engineering (Sanjivani College of Engineering)')
st.title('Name of the Students')
st.write('Ansari Mohammed Anas PRN.No. UCS20M1011')
st.write('Adarsh Borde PRN.No. UCS20M1028')
st.write('Pranav Joshi PRN.No. UCS20M1057')
st.write('Sanskruti Kekan PRN.No. UCS20M1068')
st.title('Guided by Dr.P.N.Kalavadekar')
st.title('Sugarcane Disease Detection')
st.write("Just upload your Plant's Leaf Image and get predictions if the plant is healthy or not")

# Setting the files that can be uploaded
uploaded_file = st.file_uploader("Choose a Image file", type=["png", "jpg"])

# If there is a uploaded file, start making prediction
if uploaded_file != None:
    
    # Display progress and text
    progress = st.text("Crunching Image")
    my_bar = st.progress(0)
    i = 0
    
    # Reading the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(np.array(image.resize((700, 400), resample=Image.LANCZOS)), width=None)
    my_bar.progress(i + 40)
    
    # Cleaning the image
    image = clean_image(image)
    
    # Making the predictions
    predictions, predictions_arr = get_prediction(model, image)
    my_bar.progress(i + 30)
    
    # Making the results
    result = make_results(predictions, predictions_arr)
    
    # Removing progress bar and text after prediction done
    my_bar.progress(i + 30)
    progress.empty()
    i = 0
    my_bar.empty()
    
    # Show the results
    st.write(f"The plant {result['status']} with {result['prediction']} prediction.")

    
        
