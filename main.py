import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageDraw


st.markdown("<h1 style='text-align: center; color: white;'>Plant Health Detector</h1>", unsafe_allow_html=True)
model = load_model('Saved_model.h5')
def load_image(image_file):
    img = Image.open(image_file)
    return img

nav = st.sidebar.radio("Navigation", ["HOME", "CHECK YOUR PLANT"])

if nav == "HOME":
    st.header("Identification of healthiness of plant is the key to preventing the losses in the yield and quantity of the agricultural product. It requires tremendous amount of work, expertize in the plant diseases, and also require the excessive processing time. Hence, Deep learning is used for the detection of plant healthiness. Health detection involves the steps like image acquisition, image pre-processing, image augmentation and classification.")
    st.subheader("This video shows how plant detection is done withot Deep Learning in the Hermiston Agricultural Research and Extension Center in Hermiston, Oregon")
    video_file = open('beautify//plant_video.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

elif nav == "CHECK YOUR PLANT":
    st.header("TRY OUT OUR MODEL HERE")
    image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
    if image_file is not None:
        img = load_image(image_file)
        with open(os.path.join("tempDir",image_file.name),"wb") as f: 
          f.write(image_file.getbuffer())
        test_image = load_img(os.path.join("tempDir",image_file.name), target_size = (64, 64))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        if result[0][0] == 1:
            st.markdown("<h2 style='text-align: center; color: red;'>Plant is Unhealthy</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: red;'>Plant is Healhty</h2>", unsafe_allow_html=True)
        

        
