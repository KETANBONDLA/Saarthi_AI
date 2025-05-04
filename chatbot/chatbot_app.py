import os
import dotenv
import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
dotenv.load_dotenv()

# Configuration and setup
genai.configure(api_key=os.getenv("GENAI_KEY"))  # Replace with your actual Gemini API key
model = genai.GenerativeModel("gemini-1.5-pro-001")

# Language support dictionary
language_codes = {
    "हिंदी (Hindi)": "hi",
    "English": "en",
    "বাংলা (Bengali)": "bn",
    "తెలుగు (Telugu)": "te",
    "मराठी (Marathi)": "mr",
    "தமிழ் (Tamil)": "ta",
    "ગુજરાતી (Gujarati)": "gu",
    "ಕನ್ನಡ (Kannada)": "kn",
    "ਪੰਜਾਬੀ (Punjabi)": "pa",
    "ଓଡ଼ିଆ (Odia)": "or"
}

# Speech to text conversion based on selected language
def speech_to_text(language_code):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        st.info(f"🎤 Listening... (Please speak now)")
        try: 
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=50)
            try:
                text = recognizer.recognize_google(audio, language=language_code)
                return text
            except sr.UnknownValueError:
                return "❌ Could not understand audio"
            except sr.RequestError:
                return "❌ Speech service unavailable"
        except sr.WaitTimeoutError:
            return "❌ No speech detected within timeout period"


def preprocess_text(text, language_code):
    """Apply NLP preprocessing techniques to user input"""
    # Convert to lowercase (for non-Hindi languages)
    if language_code != "hi":
        text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Basic stopword removal for English
    if language_code == "en":
        common_stop_words = ['the', 'a', 'an', 'is', 'are', 'in', 'at', 'on']
        words = text.split()
        filtered_words = [word for word in words if word not in common_stop_words]
        text = ' '.join(filtered_words)
    
    # Log preprocessing steps for demonstration
    st.session_state['preprocessing_log'] = f"Original: '{text}'\nPreprocessed: '{text}'"
    
    return text


def classify_intent(question, language_code):
    """Simple rule-based intent classification"""
    # Define keywords for different intents in different languages
    intent_keywords = {
        "disease": {
            "en": ["disease", "infected", "spots", "wilting", "yellow", "pest"],
            "hi": ["रोग", "कीट", "बीमारी", "धब्बे", "सूखना", "पीला"]
        },
        "fertilizer": {
            "en": ["fertilizer", "nutrient", "npk", "manure", "compost"],
            "hi": ["उर्वरक", "खाद", "पोषक तत्व", "नत्रजन", "फास्फोरस"]
        },
        "weather": {
            "en": ["rain", "weather", "temperature", "monsoon", "irrigation"],
            "hi": ["बारिश", "मौसम", "तापमान", "वर्षा", "सिंचाई"]
        },
        "market": {
            "en": ["price", "market", "sell", "cost", "profit", "msp"],
            "hi": ["मूल्य", "बाजार", "विक्रय", "लागत", "लाभ"]
        }
    }
    
    question = question.lower()
    detected_intents = []
    
    # Check for each intent
    for intent, keywords in intent_keywords.items():
        lang_keys = keywords.get(language_code, keywords.get("en", []))
        for keyword in lang_keys:
            if keyword.lower() in question:
                detected_intents.append(intent)
                break
    
    # Default intent if none detected
    if not detected_intents:
        return "general"
    
    # Return the most likely intent
    return detected_intents[0]

def extract_entities(text, language_code):
    """Simple rule-based named entity recognition for crops and locations"""
    # Sample crop and location dictionaries (expand these)
    crops = {
        "en": ["rice", "wheat", "maize", "cotton", "sugarcane", "potato", "tomato"],
        "hi": ["चावल", "गेहूँ", "मक्का", "कपास", "गन्ना", "आलू", "टमाटर"]
    }
    
    indian_states = {
        "en": ["maharashtra", "punjab", "uttar pradesh", "karnataka", "gujarat"],
        "hi": ["महाराष्ट्र", "पंजाब", "उत्तर प्रदेश", "कर्नाटक", "गुजरात"]
    }
    
    found_crops = []
    found_locations = []
    
    text_lower = text.lower()
    
    # Check for crops
    crop_list = crops.get(language_code, crops["en"])
    for crop in crop_list:
        if crop.lower() in text_lower:
            found_crops.append(crop)
    
    # Check for locations
    location_list = indian_states.get(language_code, indian_states["en"])
    for location in location_list:
        if location.lower() in text_lower:
            found_locations.append(location)
    
    return {
        "crops": found_crops,
        "locations": found_locations
    }

def analyze_sentiment(text, language_code):
    """Simple rule-based sentiment analysis"""
    # Positive and negative keywords in different languages
    sentiments = {
        "positive": {
            "en": ["good", "better", "best", "increase", "improve", "help", "benefit"],
            "hi": ["अच्छा", "बेहतर", "बढ़िया", "वृद्धि", "सुधार", "मदद", "लाभ"]
        },
        "negative": {
            "en": ["bad", "worse", "decrease", "damage", "problem", "issue", "worry"],
            "hi": ["बुरा", "खराब", "कमी", "नुकसान", "समस्या", "चिंता", "हानि"]
        }
    }
    
    text_lower = text.lower()
    pos_count = 0
    neg_count = 0
    
    # Count sentiment words
    for word in sentiments["positive"].get(language_code, sentiments["positive"]["en"]):
        if word in text_lower:
            pos_count += 1
    
    for word in sentiments["negative"].get(language_code, sentiments["negative"]["en"]):
        if word in text_lower:
            neg_count += 1
    
    # Determine sentiment
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"
    
def generate_response_template(intent, entities, sentiment, language_code):
    """Generate response template based on NLP analysis"""
    templates = {
        "disease": {
            "en": "I see you're asking about plant diseases{crop_info}. Let me provide information on that...",
            "hi": "मैं देख रहा हूँ कि आप पौधों के रोगों{crop_info} के बारे में पूछ रहे हैं। मुझे इस पर जानकारी प्रदान करें..."
        },
        "fertilizer": {
            "en": "You're interested in fertilizers{crop_info}. Here's what you should know...",
            "hi": "आप उर्वरकों{crop_info} में रुचि रखते हैं। आपको यह जानना चाहिए..."
        },
        # Add more templates for other intents
    }
    
    # Insert crop information if available
    crop_info = ""
    if entities["crops"]:
        if language_code == "en":
            crop_info = f" for {', '.join(entities['crops'])}"
        elif language_code == "hi":
            crop_info = f" {', '.join(entities['crops'])} के लिए"
    
    # Get appropriate template
    template = templates.get(intent, {}).get(language_code, "I'll help you with that question.")
    if isinstance(template, str):
        template = template.format(crop_info=crop_info)
    
    return template

# Generate language-specific greeting
def get_greeting(language_code):
    greetings = {
        "hi": "नमस्ते, मैं सारथी हूँ। आप कृषि से संबंधित कोई भी प्रश्न पूछ सकते हैं।",
        "en": "Hello, I am Saarthi. You can ask me any agriculture-related question.",
        "bn": "নমস্কার, আমি সারথি। আপনি আমাকে কৃষি সম্পর্কিত যেকোন প্রশ্ন জিজ্ঞাসা করতে পারেন।",
        "te": "హలో, నేను సారథి. మీరు వ్యవసాయానికి సంబంధించిన ఏ ప్రశ్నను అడగవచ్చు.",
        "mr": "नमस्कार, मी सारथी आहे. तुम्ही मला शेतीसंबंधित कोणताही प्रश्न विचारू शकता.",
        "ta": "வணக்கம், நான் சாரதி. நீங்கள் விவசாயம் தொடர்பான எந்தக் கேள்வியையும் என்னிடம் கேட்கலாம்.",
        "gu": "નમસ્તે, હું સારથી છું. તમે મને કૃષિ સંબંધિત કોઈપણ પ્રશ્ન પૂછી શકો છો.",
        "kn": "ನಮಸ್ಕಾರ, ನಾನು ಸಾರಥಿ. ನೀವು ನನ್ನನ್ನು ಕೃಷಿಗೆ ಸಂಬಂಧಿಸಿದ ಯಾವುದೇ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಬಹುದು.",
        "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਮੈਂ ਸਾਰਥੀ ਹਾਂ। ਤੁਸੀਂ ਮੈਨੂੰ ਖੇਤੀਬਾੜੀ ਨਾਲ ਸਬੰਧਤ ਕੋਈ ਵੀ ਸਵਾਲ ਪੁੱਛ ਸਕਦੇ ਹੋ।",
        "or": "ନମସ୍କାର, ମୁଁ ସାରଥୀ। ଆପଣ ମୋତେ କୃଷି ସମ୍ବନ୍ଧୀୟ କୌଣସି ପ୍ରଶ୍ନ ପଚାରି ପାରିବେ।"
    }
    return greetings.get(language_code, greetings["en"])

# Ask Gemini model with language-specific prompt
def ask_gemini(question, language_code, intent=None, entities=None):
    # Create language-specific prompt
    language_prompts = {
        "hi": f"एक कृषि विशेषज्ञ की तरह, सरल हिंदी में उत्तर दें। उत्तर विस्तृत और सटीक होना चाहिए।\n\nप्रश्न: {question}",
        "en": f"As an agricultural expert, answer in simple English. The answer should be detailed and accurate.\n\nQuestion: {question}",
        "bn": f"একজন কৃষি বিশেষজ্ঞ হিসাবে, সহজ বাংলায় উত্তর দিন। উত্তর বিস্তারিত এবং সঠিক হওয়া উচিত।\n\nপ্রশ্ন: {question}",
        "te": f"వ్యవసాయ నిపుణుడిగా, సరళమైన తెలుగులో సమాధానం ఇవ్వండి. సమాధానం వివరణాత్మకంగా మరియు ఖచ్చితంగా ఉండాలి।\n\nప్రశ్న: {question}",
        "mr": f"कृषी तज्ञ म्हणून, सोप्या मराठीत उत्तर द्या. उत्तर तपशीलवार आणि अचूक असावे।\n\nप्रश्न: {question}",
        "ta": f"வேளாண் நிபுணராக, எளிய தமிழில் பதிலளிக்கவும். பதில் விரிவானதாகவும் துல்லியமானதாகவும் இருக்க வேண்டும்।\n\nகேள்வி: {question}",
        "gu": f"કૃષિ નિષ્ણાત તરીકે, સરળ ગુજરાતીમાં જવાબ આપો. જવાબ વિગતવાર અને સચોટ હોવો જોઈએ।\n\nપ્રશ્ન: {question}",
        "kn": f"ಕೃಷಿ ತಜ್ಞರಾಗಿ, ಸರಳ ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ. ಉತ್ತರವು ವಿವರವಾಗಿ ಮತ್ತು ನಿಖರವಾಗಿರಬೇಕು।\n\nಪ್ರಶ್ನೆ: {question}",
        "pa": f"ਖੇਤੀਬਾੜੀ ਮਾਹਿਰ ਵਜੋਂ, ਸਧਾਰਨ ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦਿਓ। ਜਵਾਬ ਵਿਸਥਾਰਪੂਰਵਕ ਅਤੇ ਸਹੀ ਹੋਣਾ ਚਾਹੀਦਾ ਹੈ।\n\nਸਵਾਲ: {question}",
        "or": f"ଜଣେ କୃଷି ବିଶେଷଜ୍ଞ ଭାବରେ, ସରଳ ଓଡ଼ିଆରେ ଉତ୍ତର ଦିଅନ୍ତୁ। ଉତ୍ତର ବିସ୍ତୃତ ଓ ସଠିକ୍ ହେବା ଉଚିତ।\n\nପ୍ରଶ୍ନ: {question}"
    }
    
    # Enhance prompt with NLP insights
    nlp_context = ""
    if intent:
        nlp_context += f"\nThe user is asking about {intent}. "
    
    if entities and entities["crops"]:
        nlp_context += f"\nMentioned crops: {', '.join(entities['crops'])}. "
    
    if entities and entities["locations"]:
        nlp_context += f"\nMentioned locations: {', '.join(entities['locations'])}. "
    
    # Combine everything
    prompt = f"{language_prompts.get(language_code, language_prompts['en'])}{nlp_context}\n\nQuestion: {question}"
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"
    

# Enhanced styling with professional touch
st.markdown("""
    <style>
        .saarthi-title {
            font-size: 50px;
            font-weight: 800;
            color: #2E7D32;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 20px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .saarthi-subtitle {
            font-size: 24px;
            font-weight: 500;
            color: #555555;
            text-align: center;
            margin-bottom: 30px;
        }
        .question-container {
            font-size: 18px;
            line-height: 1.6;
            color: #2c3e50;
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #3498db;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .answer-container {
            font-size: 18px;
            line-height: 1.6;
            color: #2c3e50;
            background-color: #f0f7fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #27ae60;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 50px;
            padding: 15px 32px;
            font-size: 22px;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            width: auto;
            margin: 20px auto;
            display: block;
        }
        .stButton>button:hover {
            background-color: #2980b9;
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
        .info-box {
            padding: 12px;
            border-radius: 8px;
            font-size: 18px;
        }
        .success-header {
            font-size: 22px;
            font-weight: 600;
            color: #27ae60;
            margin: 20px 0 10px 0;
        }
        .language-selector {
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .feature-badge {
            background-color: #f0f7fa;
            border-radius: 20px;
            padding: 5px 15px;
            font-size: 12px;
            color: #2980b9;
            margin-right: 10px;
            border: 1px solid #bde0f3;
        }
    </style>
""", unsafe_allow_html=True)

def run():
    # App Title and Description with enhanced styling
    st.markdown('<div class="saarthi-title">🌾 सारथी AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="saarthi-subtitle">Your Multilingual Agricultural Assistant</div>', unsafe_allow_html=True)
    
    # Language selection
    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    selected_language = st.selectbox(
        "Select Language / भाषा चुनें",
        list(language_codes.keys()),
        index=0
    )
    language_code = language_codes[selected_language]
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display greeting in selected language
    st.info(get_greeting(language_code))
    
    # Add a container for better visual separation
    with st.container():
        # Voice input option
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Speech Input Button
            if st.button(f"🎙️ Ask Question by Voice"):
                user_question = speech_to_text(language_code)
                if user_question:
                    st.session_state['user_question'] = user_question
                    
        with col2:
            # Text input option
            text_question = st.text_input("Or type your question here:", 
                                     key="text_input",
                                     placeholder=f"Enter your question in {selected_language}")
            if text_question:
                st.session_state['user_question'] = text_question
        
        # Process the question (either from voice or text input)
        # Process the question (either from voice or text input)
        if 'user_question' in st.session_state and st.session_state['user_question']:
            user_question = st.session_state['user_question']
            
            # Display the original question
            st.markdown(f'<div class="question-container"><strong>❓ Question:</strong> {user_question}</div>', unsafe_allow_html=True)
            
            if "❌" not in user_question:
                # Apply NLP techniques with a collapsible section to show the process
                with st.spinner("🤖 Analyzing and processing your question..."):
                    # Preprocess text
                    processed_text = preprocess_text(user_question, language_code)
                    
                    # Extract intent
                    intent = classify_intent(processed_text, language_code)
                    
                    # Extract entities
                    entities = extract_entities(processed_text, language_code)
                    
                    # Analyze sentiment
                    sentiment = analyze_sentiment(processed_text, language_code)
                    
                    # Generate response template
                    template = generate_response_template(intent, entities, sentiment, language_code)
                    
                    # Get complete answer from Gemini
                    answer = ask_gemini(user_question, language_code, intent=intent, entities=entities)
                    
                    # Display NLP analysis (optional, can be hidden behind an expander)
                    with st.expander("See NLP Analysis", expanded=False):
                        st.markdown("### NLP Pipeline Results")
                        st.markdown(f"**Intent Classification:** {intent}")
                        st.markdown(f"**Entities Recognized:**")
                        st.markdown(f"- Crops: {', '.join(entities['crops']) if entities['crops'] else 'None'}")
                        st.markdown(f"- Locations: {', '.join(entities['locations']) if entities['locations'] else 'None'}")
                        st.markdown(f"**Sentiment Analysis:** {sentiment}")
                        st.markdown(f"**Response Template:** {template}")
                    
                    # Display the final answer
                    st.markdown('<div class="success-header">✅ Answer:</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-container">{answer}</div>', unsafe_allow_html=True)

