# 🌾 Saarthi-AI: Multilingual Agricultural Assistant

**Saarthi-AI** is a multilingual, voice-enabled agricultural assistant developed to empower Indian farmers by providing timely, accurate, and accessible information on farming practices, crop diseases, and yield predictions. Designed with inclusivity at its core, Saarthi-AI breaks language and literacy barriers using cutting-edge AI and NLP technologies.

---

## 🔗 Table of Contents
- [🚜 Chatbot Module](#-chatbot-module)
- [🦠 Disease Detection Module](#-disease-detection-module)
- [📈 Yield Prediction Module](#-yield-prediction-module)
- [🛠️ Tech Stack](#-tech-stack)
- [📌 Future Scope](#-future-scope)
- [📄 License](#-license)

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
- TensorFlow or PyTorch for deep learning models
- Streamlit or Flask for deployment
- PlantVillage or custom datasets for training

---

## 📈 Yield Prediction Module

The **Yield Prediction** module uses historical and real-time data (soil, rainfall, seed type) to forecast crop yields.

### 🎯 Goals
- Predict expected output per hectare
- Provide suggestions on seed selection and irrigation schedules
- Assist in financial planning for small-scale farmers

### 🧰 Technologies
- Random Forest or XGBoost for prediction models
- Scikit-learn for feature engineering
- Matplotlib for data visualization

---

## 🛠️ Tech Stack Summary

- **Chatbot**: Streamlit, SpeechRecognition, Google Gemini 1.5 Pro, Python
- **Disease Detection**: OpenCV, TensorFlow, Streamlit
- **Yield Prediction**: Scikit-learn, XGBoost

---

## 📌 Future Scope

- Offline functionality for rural deployment
- Image recognition for real-time disease detection
- IoT sensor integration for precision agriculture
- Addition of tribal languages and dialects
- Integration with government schemes and market linkages

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🙌 Authors

- **Ketan Bondla** (2022-B-17122003)
- **Rudhi Pareek** (2022-B-16012004B)
- **Vaikhari Kanetkar** (2022-B-15092004B)

---

## 📬 Feedback

Feel free to open issues or contact us for collaboration, contributions, or deployment inquiries.
