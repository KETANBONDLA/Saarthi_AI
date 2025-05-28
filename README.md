# 🌾 Saarthi-AI: Multilingual Agricultural Assistant

**Saarthi-AI** is a multilingual, voice-enabled agricultural assistant developed to empower Indian farmers by providing timely, accurate, and accessible information on farming practices, crop diseases, and yield predictions. Designed with inclusivity at its core, Saarthi-AI breaks language and literacy barriers using cutting-edge AI and NLP technologies.

---

## 🔗 Table of Contents
- [🚜 Chatbot Module](#-chatbot-module)
- [🦠 Disease Detection Module](#-disease-detection-module)
- [📈 Yield Prediction Module](#-yield-prediction-module)
- [🗃️ Datasets Used](#-datasets-used)
- [📌 Future Scope](#-future-scope)

---

## 🚜 Chatbot Module

The **Saarthi-AI Chatbot** is a multilingual voice assistant designed to answer agricultural queries in 10 major Indian languages, including Hindi, Marathi, Tamil, Bengali, and more.

### ✨ Features
- Voice-enabled queries for accessibility in low-literacy regions
- Supports 10 Indian languages using ISO scripts
- Answers about weather, fertilizers, pest control, market prices, and more
- Region-specific and crop-specific recommendations
- Sentiment-aware replies for user engagement

### 🧠 NLP Pipeline
- Text Preprocessing: Language-aware normalization and stopword removal
- Intent Recognition: Classification of user intents (e.g., market, disease)
- Entity Extraction: Crop, location, and seasonal context extraction
- Sentiment Analysis: Contextual understanding to refine tone

### 🧰 Technologies Used
- Streamlit for interface development
- SpeechRecognition for multilingual voice processing
- Google Gemini 1.5 Pro as the foundational language model
- Python with custom NLP pipelines

---

## 🦠 Disease Detection Module 

The **Disease Detection** submodule aims to integrate computer vision and deep learning to identify common crop diseases from images captured via mobile phones.

### 🎯 Goals
- Detect diseases like blight, mildew, rust, etc.
- Suggest organic and chemical treatment methods
- Enable photo-based detection via mobile

### 🧰 Technologies
- OpenCV for image preprocessing
- TensorFlow for deep learning models
- Streamlit for deployment
- PlantVillage datasets for training

---

## 📈 Yield Prediction Module

The **Yield Prediction** module uses historical and real-time data (soil, rainfall, seed type) to forecast crop yields.

### 🎯 Goals
- Predict expected output per hectare
- Provide suggestions on seed selection and irrigation schedules
- Assist in financial planning for small-scale farmers

### 🧰 Technologies
- Random Forest for prediction models
- Scikit-learn for feature engineering
- Matplotlib for data visualization

---

## 🗃️ Datasets Used

### 🌿 Disease Detection
- **[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)**  
  A diverse dataset containing labeled images of healthy and diseased plant leaves across various crop types.

### 🌾 Yield Prediction
- **[Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)**  
  Includes historical data on temperature, rainfall, and production for major crops across Indian states.

---

## 📌 Future Scope

- Offline functionality for rural deployment
- Image recognition for real-time disease detection
- IoT sensor integration for precision agriculture
- Addition of tribal languages and dialects
- Integration with government schemes and market linkages

---
## 🙌 Contributors

- **Ketan Bondla**
- **Rudhi Pareek**
- **Vaikhari Kanetkar** 

---

## 📬 Feedback

Feel free to open issues or contact us for collaboration, contributions, or deployment inquiries.
