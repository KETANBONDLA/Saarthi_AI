import streamlit as st
import tensorflow as tf
import numpy as np
import os

#Tensorflow Model Prediction
def model_prediction(test_image):
    model_path = os.path.join('disease_detection', 'rebuilt_model.h5')
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

def run():
    st.header("🔍 Plant Disease Recognition System")
    
    # Main content - no sidebar
    st.markdown("""
    Upload an image of a plant leaf to detect diseases. Our system uses machine learning to identify 
    potential plant diseases across 38 different categories.
    """)
    
    # Create two tabs for the disease app
    disease_tab1, disease_tab2 = st.tabs(["Disease Detection", "About Dataset"])
    
    with disease_tab1:
        st.subheader("Upload Plant Image")
        test_image = st.file_uploader("Choose a leaf image:", type=['jpg', 'jpeg', 'png'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if test_image is not None:
                st.image(test_image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if test_image is not None:
                if st.button("Detect Disease"):
                    with st.spinner("Analyzing image..."):
                        result_index = model_prediction(test_image)
                        
                        #Define Class
                        class_name = ['Apple___Apple_scab',
                        'Apple___Black_rot',
                        'Apple___Cedar_apple_rust',
                        'Apple___healthy',
                        'Blueberry___healthy',
                        'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy',
                        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_',
                        'Corn_(maize)___Northern_Leaf_Blight',
                        'Corn_(maize)___healthy',
                        'Grape___Black_rot',
                        'Grape___Esca_(Black_Measles)',
                        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy',
                        'Orange___Haunglongbing_(Citrus_greening)',
                        'Peach___Bacterial_spot',
                        'Peach___healthy',
                        'Pepper,_bell___Bacterial_spot',
                        'Pepper,_bell___healthy',
                        'Potato___Early_blight',
                        'Potato___Late_blight',
                        'Potato___healthy',
                        'Raspberry___healthy',
                        'Soybean___healthy',
                        'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch',
                        'Strawberry___healthy',
                        'Tomato___Bacterial_spot',
                        'Tomato___Early_blight',
                        'Tomato___Late_blight',
                        'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot',
                        'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot',
                        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                        'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
                        
                        predicted_class = class_name[result_index]
                        st.success(f"Diagnosis: {predicted_class.replace('___', ' - ')}")
                        
                        # Format result for better readability
                        crop, condition = predicted_class.split('___')
                        st.markdown(f"""
                        **Crop**: {crop}  
                        **Condition**: {condition.replace('_', ' ')}
                        """)
    
    with disease_tab2:
        st.subheader("About the Disease Detection System")
        st.markdown("""
        #### Dataset Information
        This plant disease recognition system uses a dataset consisting of about 87,000 RGB images 
        of healthy and diseased crop leaves categorized into 38 different classes.
        
        The model was trained on 70,295 images and validated on 17,572 images to ensure accuracy 
        in detecting various plant diseases.
        
        #### Supported Plants
        The system can identify diseases in various crops including Apple, Blueberry, Cherry, Corn,
        Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
        """)