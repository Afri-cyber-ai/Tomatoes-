import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
import random

# -------------------------------
# Custom CSS for Professional Styling
# -------------------------------

st.markdown("""
<style>
/* Main purple theme /
.main-header {
font-size: 3rem;
color: #6a0dad;
font-weight: 700;
margin-bottom: 0;
}
.sub-header {
font-size: 1.5rem;
color: #8a2be2;
font-weight: 400;
margin-top: 0;
}
.agrinova-badge {
background-color: #6a0dad;
color: white;
padding: 0.5rem 1rem;
border-radius: 0.5rem;
font-weight: 600;
margin-bottom: 1rem;
display: inline-block;
}
.stButton>button {
background-color: #6a0dad;
color: white;
border: none;
padding: 0.5rem 1rem;
border-radius: 0.5rem;
font-weight: 600;
}
.stButton>button:hover {
background-color: #8a2be2;
color: white;
}
.consult-button {
background-color: #28a745;
color: white;
border: none;
padding: 0.75rem 1.5rem;
border-radius: 0.5rem;
font-weight: 700;
font-size: 1.1rem;
margin: 2rem auto;
display: block;
width: 80%;
text-align: center;
}
.consult-button:hover {
background-color: #218838;
color: white;
}
.prediction-card {
background-color: #f8f5ff;
padding: 1.5rem;
border-radius: 0.5rem;
border-left: 5px solid #6a0dad;
margin-top: 1rem;
}
.confidence-bar {
background-color: #e6e6fa;
height: 1.5rem;
border-radius: 0.3rem;
margin: 0.5rem 0;
overflow: hidden;
}
.confidence-fill {
background-color: #6a0dad;
height: 100%;
border-radius: 0.3rem;
display: flex;
align-items: center;
justify-content: flex-end;
padding-right: 0.5rem;
color: white;
font-weight: 500;
font-size: 0.8rem;
}
.footer {
text-align: center;
margin-top: 3rem;
color: #6a0dad;
font-size: 0.9rem;
}
/ Progress circle styling /
.progress-container {
display: flex;
justify-content: center;
align-items: center;
margin: 20px 0;
}
.progress-circle {
width: 100px;
height: 100px;
border-radius: 50%;
background: conic-gradient(#6a0dad 0% var(--progress), #f0f0f0 var(--progress) 100%);
display: flex;
justify-content: center;
align-items: center;
position: relative;
}
.progress-circle::before {
content: '';
position: absolute;
width: 80px;
height: 80px;
border-radius: 50%;
background: white;
}
.progress-text {
position: relative;
font-weight: bold;
color: #6a0dad;
}
.symptom-item {
margin-bottom: 8px;
padding-left: 15px;
position: relative;
}
.symptom-item:before {
content: "•";
position: absolute;
left: 0;
color: #6a0dad;
}
.disease-name {
font-size: 1.5rem;
font-weight: bold;
margin-bottom: 5px;
}
.disease-name-red {
color: #dc3545; / Red for diseases /
}
.disease-name-green {
color: #28a745; / Green for healthy /
}
.causative-agent {
font-size: 1.1rem;
color: #6a0dad;
margin-bottom: 15px;
font-weight: 600;
background-color: #f0e6ff;
padding: 8px 12px;
border-radius: 5px;
border-left: 4px solid #6a0dad;
}
.confidence {
font-size: 1.2rem;
font-weight: bold;
margin-bottom: 15px;
color: #2c662d;
}
/ Hide unnecessary Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 1. Load Local Model (from repo)
# -------------------------------

MODEL_PATH = "tomato_best_final_model.keras"  # CHANGED: Model filename updated

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# -------------------------------
# 2. Class Names and Disease Information
# -------------------------------

# CHANGED: Updated class names for tomato diseases
CLASS_NAMES = [
    "Tomato Bacterial spot",
    "Tomato Early blight",
    "Tomato Late blight",
    "Tomato Leaf Mold",
    "Tomato Septoria leaf spot",
    "Tomato Spider mites Two-spotted spider mite",
    "Tomato Target Spot",
    "Tomato Tomato YellowLeaf Curl Virus",
    "Tomato Tomato mosaic virus",
    "Tomato healthy"
]

# CHANGED: Replaced entire disease_info dictionary with tomato disease data
disease_info = {
    "Tomato Bacterial spot": {
        "causative_agent": "Bacterium Xanthomonas campestris pv. vesicatoria",
        "symptoms": [
            "Small, water-soaked spots on leaves that turn dark brown or black",
            "Spots may have a yellow halo and can merge to form larger blighted areas",
            "Lesions on stems and fruits, often with a raised, scabby appearance"
        ],
        "prevention": [
            "Use disease-free seeds and transplants",
            "Practice crop rotation with non-host plants",
            "Avoid overhead irrigation to reduce leaf wetness",
            "Apply copper-based bactericides preventatively"
        ],
        "confidence": 0.92
    },
    "Tomato Early blight": {
        "causative_agent": "Fungus Alternaria solani",
        "symptoms": [
            "Dark brown to black concentric rings on lower leaves resembling bull's-eye",
            "Yellowing and wilting of leaves starting from the bottom of the plant",
            "Lesions on stems and fruits, leading to fruit rot"
        ],
        "prevention": [
            "Plant resistant varieties if available",
            "Stake plants to improve air circulation",
            "Remove and destroy infected plant debris",
            "Apply fungicides containing chlorothalonil or mancozeb"
        ],
        "confidence": 0.88
    },
    "Tomato Late blight": {
        "causative_agent": "Oomycete Phytophthora infestans",
        "symptoms": [
            "Large, water-soaked, gray-green lesions on leaves that rapidly expand",
            "White, fuzzy fungal growth on undersides of leaves under humid conditions",
            "Dark, firm rot on fruits and stems, leading to complete plant collapse"
        ],
        "prevention": [
            "Avoid planting in poorly drained areas",
            "Use drip irrigation instead of overhead watering",
            "Apply fungicides containing mefenoxam or chlorothalonil preventatively",
            "Remove and destroy infected plants immediately"
        ],
        "confidence": 0.90
    },
    "Tomato Leaf Mold": {
        "causative_agent": "Fungus Passalora fulva (formerly Fulvia fulva)",
        "symptoms": [
            "Pale green to yellow spots on upper leaf surfaces",
            "Velvety, olive-green to brown mold on lower leaf surfaces",
            "Leaves curl and die, starting from the lower parts of the plant"
        ],
        "prevention": [
            "Maintain low humidity through proper spacing and ventilation",
            "Water plants at the base to keep foliage dry",
            "Apply fungicides such as chlorothalonil or copper-based products",
            "Remove and destroy crop residues after harvest"
        ],
        "confidence": 0.85
    },
    "Tomato Septoria leaf spot": {
        "causative_agent": "Fungus Septoria lycopersici",
        "symptoms": [
            "Small, circular spots with dark brown margins and light gray centers",
            "Tiny black specks (fungal fruiting bodies) within the spots",
            "Severe defoliation starting from the bottom of the plant"
        ],
        "prevention": [
            "Rotate crops with non-solanaceous plants for at least 2 years",
            "Avoid working with plants when they are wet",
            "Use mulch to prevent soil splashing onto leaves",
            "Apply fungicides like mancozeb or copper-based products"
        ],
        "confidence": 0.87
    },
    "Tomato Spider mites Two-spotted spider mite": {
        "causative_agent": "Arachnid Tetranychus urticae",
        "symptoms": [
            "Fine stippling or yellow speckling on leaves",
            "Leaves may turn bronze, curl, and drop prematurely",
            "Fine webbing on undersides of leaves or between stems"
        ],
        "prevention": [
            "Monitor plants regularly for early detection",
            "Use insecticidal soaps or horticultural oils",
            "Encourage natural predators like ladybugs and lacewings",
            "Avoid broad-spectrum insecticides that kill beneficial insects"
        ],
        "confidence": 0.83
    },
    "Tomato Target Spot": {
        "causative_agent": "Fungus Corynespora cassiicola",
        "symptoms": [
            "Brown spots with concentric rings (target-like appearance)",
            "Spots can enlarge and cause leaf yellowing and defoliation",
            "Lesions on stems and fruits, leading to fruit rot"
        ],
        "prevention": [
            "Use pathogen-free seeds and transplants",
            "Practice good sanitation by removing plant debris",
            "Apply fungicides such as chlorothalonil or azoxystrobin",
            "Avoid overhead irrigation to reduce leaf wetness"
        ],
        "confidence": 0.84
    },
    "Tomato Tomato YellowLeaf Curl Virus": {
        "causative_agent": "Virus Tomato yellow leaf curl virus (TYLCV)",
        "symptoms": [
            "Upward curling of leaves, especially younger ones",
            "Yellowing (chlorosis) of leaf margins",
            "Stunted growth and reduced fruit production"
        ],
        "prevention": [
            "Use resistant varieties when available",
            "Control whitefly populations with insecticides or reflective mulches",
            "Remove and destroy infected plants to reduce virus source",
            "Use virus-free transplants"
        ],
        "confidence": 0.89
    },
    "Tomato Tomato mosaic virus": {
        "causative_agent": "Virus Tomato mosaic virus (ToMV)",
        "symptoms": [
            "Light and dark green mottling or mosaic patterns on leaves",
            "Leaf distortion, blistering, and fern-like appearance",
            "Stunted growth and reduced yield"
        ],
        "prevention": [
            "Use virus-free seeds and transplants",
            "Practice good hygiene (wash hands, disinfect tools)",
            "Control weed hosts that may harbor the virus",
            "Remove and destroy infected plants promptly"
        ],
        "confidence": 0.86
    },
    "Tomato healthy": {
        "causative_agent": "None",
        "symptoms": [
            "Leaves are uniformly green without spots or discoloration",
            "Normal plant growth and development",
            "No signs of pests or diseases"
        ],
        "prevention": [
            "Continue good cultural practices (proper watering, fertilization)",
            "Regularly monitor plants for early signs of problems",
            "Maintain overall plant health to resist infections"
        ],
        "confidence": 0.95
    }
}

# -------------------------------
# 3. Enhanced Streamlit Interface
# -------------------------------

st.set_page_config(
    page_title="AgriNova.ai - Tomato Disease Detection",  # CHANGED: Page title updated
    page_icon="🍅",  # CHANGED: Icon updated to tomato
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header Section
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="main-header">AgriNova.ai</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Intelligent Crop Disease Detection</h2>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="agrinova-badge">Tomato Disease Classifier</div>', unsafe_allow_html=True)  # CHANGED: Badge text updated

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Main content
if model is None:
    st.error("Model loading failed. Please check if the model file exists in the repository.")
    st.info("""
    Troubleshooting steps:
    - Verify the model file 'tomato_best_final_model.keras' is in the root directory  # CHANGED: Model filename in message updated
    - Check the file is not corrupted
    - Refresh the page to try again
    """)
else:
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:  
        st.markdown("### 📤 Upload Tomato Leaf Image")  # CHANGED: Text updated
        st.markdown("Upload a clear image of a tomato leaf for disease analysis")  # CHANGED: Text updated
        
        uploaded_file = st.file_uploader(  
            "Choose an image file",   
            type=["jpg", "jpeg", "png"],  
            label_visibility="collapsed"  
        )  
        
        if uploaded_file:  
            # Show uploaded image with a border  
            image = Image.open(uploaded_file).convert("RGB")  
            st.image(image, caption="Uploaded Tomato Leaf", use_container_width=True)  # CHANGED: Caption updated
            
            if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):  
                # Simulate analysis with variable time (5-15 seconds)  
                analysis_time = random.randint(5, 15)  
                progress_placeholder = st.empty()  
                
                # Show progress circle  
                for i in range(analysis_time):  
                    progress_percent = (i + 1) / analysis_time * 100  
                    progress_placeholder.markdown(f"""  
                    <div class="progress-container">  
                        <div class="progress-circle" style="--progress: {progress_percent}%">  
                            <div class="progress-text">{int(progress_percent)}%</div>  
                        </div>  
                    </div>  
                    <p style="text-align: center;">Analyzing... {i+1}/{analysis_time} seconds</p>  
                    """, unsafe_allow_html=True)  
                    time.sleep(1)  
                
                # Clear progress indicator  
                progress_placeholder.empty()  
                
                # Preprocess image and run prediction  
                img = image.resize((224, 224))  
                img_array = np.expand_dims(np.array(img) / 255.0, axis=0)  

                # Run prediction  
                try:  
                    preds = model.predict(img_array)  
                    confidence = np.max(preds) * 100  
                    predicted_class = CLASS_NAMES[np.argmax(preds)]  
                    
                    # Store results in session state  
                    st.session_state.prediction = {  
                        "class": predicted_class,  
                        "confidence": confidence,  
                        "all_predictions": preds[0]  
                    }  
                except Exception as e:  
                    st.error(f"Error during prediction: {str(e)}")  

    with col2:  
        if uploaded_file and "prediction" in st.session_state:  
            pred = st.session_state.prediction  
            
            st.markdown("### 📊 Analysis Results")  
            
            # Results card  
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)  
            
            # Get disease data  
            disease_data = disease_info[pred["class"]]  
            
            # Display disease name with color coding  
            if pred["class"] == "Tomato healthy":  # CHANGED: Class name updated
                st.markdown(f'<div class="disease-name disease-name-green">{pred["class"]}</div>', unsafe_allow_html=True)  
            else:  
                st.markdown(f'<div class="disease-name disease-name-red">{pred["class"]}</div>', unsafe_allow_html=True)  
            
            # Display causative agent with enhanced visibility  
            st.markdown(f'<div class="causative-agent">Causative Agent: {disease_data["causative_agent"]}</div>', unsafe_allow_html=True)  
            
            # Display confidence  
            st.markdown(f'<div class="confidence">Confidence: {pred["confidence"]:.2f}%</div>', unsafe_allow_html=True)  
            
            # Display symptoms  
            st.markdown("**Typical Symptoms:**")  
            for symptom in disease_data["symptoms"]:  
                st.markdown(f'<div class="symptom-item">{symptom}</div>', unsafe_allow_html=True)  
            
            # Display prevention measures  
            st.markdown("**Prevention/Management:**")  
            for measure in disease_data["prevention"]:  
                st.markdown(f'<div class="symptom-item">{measure}</div>', unsafe_allow_html=True)  
            
            st.markdown('</div>', unsafe_allow_html=True)  
            
        else:  
            st.info("👈 Upload an image and click 'Analyze' to get results")  

    # Add "Consult an Expert" button below the two columns  
    if uploaded_file and "prediction" in st.session_state:  
        st.markdown("---")  
        col1, col2, col3 = st.columns([1, 2, 1])  
        with col2:  
            st.markdown(  
                f'<a href="tel:+2348136626696" style="text-decoration: none;">'  
                f'<button class="consult-button">📞 Speak to an Expert: +2348136626696</button>'  
                f'</a>',  
                unsafe_allow_html=True  
            )  
            
# Add some information about the system  
st.markdown("---")  
st.markdown("### ℹ️ About This System")  

info_col1, info_col2, info_col3 = st.columns(3)  

with info_col1:  
    st.markdown("**Technology**")  
    st.markdown("- Deep Learning AI")  
    st.markdown("- Computer Vision")  
    st.markdown("- TensorFlow Backend")  
    
with info_col2:  
    st.markdown("**Capabilities**")  
    st.markdown("- 10 Disease Classifications")  # CHANGED: Number updated
    st.markdown("- Real-time Analysis")  
    st.markdown("- Confidence Scoring")  
    
with info_col3:  
    st.markdown("**Benefits**")  
    st.markdown("- Early Disease Detection")  
    st.markdown("- Reduced Crop Loss")  
    st.markdown("- Increased Yield")

# Footer
st.markdown("---")
st.markdown('<div class="footer">AgriNova.ai • AI-Powered Agricultural Solutions • © 2025</div>', unsafe_allow_html=True)
