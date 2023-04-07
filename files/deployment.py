import streamlit as st
import numpy as np
import tensorflow as tf
import plotly.express as px





image =  tf.keras.preprocessing.image
load_model = tf.keras.models.load_model




st.markdown("""

<style>
    .css-9s5bis.edgvbvh3{
        display:none;
    }
    
    .css-1q1n0ol.egzxvld0
    {
        display: none;    
    }
    
</style>                       
            
            """, unsafe_allow_html=True)

st.title("Classification image ('Cat or Dog')")

with st.form("Classification image : "):
    st.subheader("Please upload your image or images .")
    st.write("then enter Classification button")
    
    selection_images = st.file_uploader("Please Upload your Image here : ", 
                             type=["jpg", "png", "jfif", "gif", "tiff", "jpeg", "HEIC"],
                             accept_multiple_files=True)
    
    
    classfic = st.form_submit_button("Classification")




#####################################################################

model = load_model(r"F:\computer vision files\files\cats_dogs_model.h5") # load Model


def plot_predict_image(selection_image): # using function to fit selection images 
                                         # to model to mack prediction 
    check = []
    label = ""
    
    img = image.load_img(selection_image, target_size=(150, 150))
    can_plot_img= image.img_to_array(img)
    img = np.expand_dims(can_plot_img, axis=0)
    img = img / 255.0
    
    for i in range(3):
        index = np.round(model.predict(img)[0][0])
        check.append(index)
    
    if sum(check) > 1:
        label = "Dog"
    
    else:
        label = "Cat"
        
    fig = px.imshow(can_plot_img, title=f"Prediction is : {label}")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig.show(), label




if classfic:
    n = 1
    for selection_image in selection_images:
        
        out = plot_predict_image(selection_image)
        
        st.write(out[0])
        
        st.success(f"The image number {n} is : {out[1]}")
        n = n + 1
    
