import streamlit as st
# Set page config once at the top - keep only here
st.set_page_config(page_title="Saarthi AI", layout="wide", page_icon="🌾")

import os
import sys
from chatbot.chatbot_app import run as run_chatbot
from yield_prediction.yield_app import run as run_yield
from disease_detection.disease_app import run as run_disease

# Add custom CSS for better visual styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0;
        padding-top: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #43A047;
        text-align: center;
        margin-bottom: 2rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    .tool-container {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stSelectbox label {
        font-size: 1.3rem;
        font-weight: 600;
        color: #33691E;
        margin-bottom: 8px;
    }
    .tool-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 5px solid #43A047;
        height: 150px;
        display: flex;
        flex-direction: column;
    }
    .tool-card h3 {
        color: #2E7D32;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }
    .tool-card p {
        font-size: 1rem;
        color: #555;
        flex-grow: 1;
    }
    .get-started {
        font-size: 1.3rem;
        color: #2E7D32;
        text-align: center;
        margin: 25px 0 10px 0;
        font-weight: 600;
    }
    /* Fix for white space by aligning containers properly */
    div.block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App Header with reduced spacing
st.markdown("<h1 class='main-header'>🌿 Saarthi AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Smart Agricultural Solutions Powered by AI</p>", unsafe_allow_html=True)

# Main content
with st.container():
    st.markdown("<div class='tool-container'>", unsafe_allow_html=True)
    
    # Dropdown menu for app selection with more compact styling
    app_choice = st.selectbox(
        "Select an AI Tool",
        ["-- Select --", "AI Chatbot", "Crop Yield Prediction", "Disease Detection"],
        index=0
    )
    
    # Display instructions based on selection
    if app_choice == "-- Select --":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="tool-card">
                <h3>💬 AI Chatbot</h3>
                <p>Get agriculture advice in Hindi language through text or voice.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="tool-card">
                <h3>📊 Crop Yield Prediction</h3>
                <p>Predict crop yields based on environmental factors.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="tool-card">
                <h3>🔍 Disease Detection</h3>
                <p>Identify plant diseases from leaf images.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<p class='get-started'>👆 Please select a tool from the dropdown above to get started.</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Load selected app
    if app_choice == "AI Chatbot":
        run_chatbot()
    elif app_choice == "Crop Yield Prediction":
        run_yield()
    elif app_choice == "Disease Detection":
        run_disease()