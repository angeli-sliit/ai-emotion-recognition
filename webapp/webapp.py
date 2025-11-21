"""
AI Emotion Recognition Web Application
Real-time facial emotion detection using deep learning.
Position your face in front of the camera for live emotion analysis.
"""

import os
import time
import sys

# Import streamlit first so we can show errors
import streamlit as st

# Try importing OpenCV with better error handling
try:
    import cv2
except ImportError as e:
    st.error(f"❌ Failed to import OpenCV: {str(e)}")
    st.error("Please ensure opencv-python-headless is installed and system dependencies are available.")
    st.error(f"Python version: {sys.version}")
    st.error(f"Python executable: {sys.executable}")
    st.stop()

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Enhanced Custom CSS - Premium Dark Modern UI
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Whole app background - Enhanced gradient */
    .stApp {
        background: radial-gradient(ellipse at top left, #1e1b4b 0%, #0f172a 25%, #020617 50%, #000000 100%);
        color: #f1f5f9;
    }

    /* Sidebar - Premium dark theme */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
        color: #f1f5f9;
        border-right: 1px solid rgba(99,102,241,0.2);
        box-shadow: 4px 0 24px rgba(0,0,0,0.5);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f1f5f9;
        text-shadow: 0 0 20px rgba(99,102,241,0.5);
    }

    .main-block {
        padding: 0 1rem 2rem 1rem;
    }

    /* --- Hero layout enhancements ---------------------------------- */
    .hero-section {
        padding: 3rem 3rem;
        background: radial-gradient(circle at top left, #4c1d95 0%, #1e1b4b 35%, #020617 100%);
        border-radius: 24px;
        margin-bottom: 2.5rem;
        box-shadow:
            0 0 0 1px rgba(129,140,248,0.45),
            0 18px 60px rgba(0,0,0,0.85),
            0 0 90px rgba(129,140,248,0.55);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(129,140,248,0.5);
    }

    .hero-section::before {
        content: '';
        position: absolute;
        inset: -40%;
        background:
            radial-gradient(circle at 10% 0%, rgba(244,114,182,0.22) 0%, transparent 45%),
            radial-gradient(circle at 90% 100%, rgba(56,189,248,0.22) 0%, transparent 50%);
        opacity: 0.9;
        pointer-events: none;
    }

    .hero-layout {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 2.5rem;
        z-index: 1;
    }

    .hero-left {
        flex: 3;
        min-width: 0;
    }

     .hero-right {
         flex: 2;
         min-width: 380px;
         max-width: 480px;
     }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.65);
        border: 1px solid rgba(244,114,182,0.7);
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #f9a8d4;
        margin-bottom: 0.7rem;
        box-shadow:
            0 0 20px rgba(244,114,182,0.55),
            inset 0 1px 0 rgba(255,255,255,0.15);
    }

    .hero-pill-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #f97316;
        box-shadow: 0 0 12px rgba(248,113,113,0.9);
    }

    .hero-title {
        font-size: 3.1rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
        background: linear-gradient(135deg, #ffffff 0%, #e5e7eb 35%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1.6px;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        max-width: 620px;
        color: #e5e7eb;
        opacity: 0.95;
        line-height: 1.7;
        margin-bottom: 1.4rem;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }

    .hero-badge {
        padding: 0.5rem 1.1rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(129,140,248,0.65);
        font-size: 0.82rem;
        font-weight: 500;
        color: #e0e7ff;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        backdrop-filter: blur(14px);
        box-shadow:
            0 0 22px rgba(129,140,248,0.45),
            inset 0 1px 0 rgba(255,255,255,0.15);
    }

    .hero-badge-icon {
        font-size: 1rem;
    }

     .hero-right-card {
         background: radial-gradient(circle at top, rgba(15,23,42,1) 0%, rgba(15,23,42,0.85) 60%, rgba(15,23,42,0.9) 100%);
         border-radius: 22px;
         padding: 2rem 1.8rem 1.6rem;
         border: 1px solid rgba(129,140,248,0.6);
         box-shadow:
             0 0 0 1px rgba(129,140,248,0.6),
             0 18px 50px rgba(15,23,42,0.95);
         position: relative;
         overflow: hidden;
         width: 100%;
     }

    .hero-right-card::before {
        content: '';
        position: absolute;
        inset: -40%;
        background:
            radial-gradient(circle at 0% 0%, rgba(129,140,248,0.25) 0%, transparent 45%),
            radial-gradient(circle at 100% 100%, rgba(250,204,21,0.22) 0%, transparent 45%);
        opacity: 0.8;
        pointer-events: none;
    }

     .hero-right-label {
         font-size: 0.95rem;
         text-transform: uppercase;
         letter-spacing: 0.14em;
         color: #a5b4fc;
         margin-bottom: 1rem;
     }

     .hero-stat-grid {
         display: grid;
         grid-template-columns: repeat(3, minmax(0, 1fr));
         gap: 1rem;
         margin-bottom: 1.2rem;
     }

     .hero-stat {
         background: rgba(15,23,42,0.9);
         border-radius: 14px;
         padding: 1rem 0.8rem;
         border: 1px solid rgba(55,65,81,0.9);
         text-align: center;
         position: relative;
         overflow: hidden;
         min-width: 0;
     }

     .hero-stat span:first-child {
         display: block;
         font-size: 0.9rem;
         color: #9ca3af;
         margin-bottom: 0.4rem;
     }

     .hero-stat span:last-child {
         display: block;
         font-size: 1.3rem;
         font-weight: 700;
         color: #e5e7eb;
     }

    .hero-stat::after {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top left, rgba(129,140,248,0.22) 0%, transparent 55%);
        opacity: 0.9;
        pointer-events: none;
    }

     .hero-caption {
         font-size: 0.88rem;
         color: #cbd5e1;
         opacity: 0.9;
         line-height: 1.5;
     }

    @media (max-width: 900px) {
        .hero-layout {
            flex-direction: column;
            align-items: flex-start;
        }
        .hero-right {
            width: 100%;
            max-width: none;
        }
    }

     /* Enhanced card styles */
     .card {
         background: linear-gradient(135deg, rgba(30,27,75,0.95) 0%, rgba(15,23,42,0.95) 100%);
         color: #f1f5f9;
         padding: 2.5rem 2.5rem;
         border-radius: 20px;
         box-shadow: 
             0 0 0 1px rgba(99,102,241,0.2),
             0 8px 32px rgba(0,0,0,0.7),
             0 0 60px rgba(99,102,241,0.2);
         border: 1px solid rgba(99,102,241,0.2);
         margin-bottom: 2rem;
         position: relative;
         overflow: hidden;
         transition: all 0.3s ease;
         width: 100%;
     }

    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at top right, rgba(99,102,241,0.15) 0%, transparent 60%);
        pointer-events: none;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 0 0 1px rgba(99,102,241,0.3),
            0 12px 40px rgba(0,0,0,0.8),
            0 0 80px rgba(99,102,241,0.3);
    }

     .card-header {
         font-weight: 700;
         font-size: 1.5rem;
         margin-bottom: 1.2rem;
         color: #f1f5f9;
         letter-spacing: -0.5px;
     }

    /* Webcam container - Enhanced (compact for screen fit) */
    .webcam-container {
        background: linear-gradient(135deg, rgba(30,27,75,0.95) 0%, rgba(15,23,42,0.95) 100%);
        color: #f1f5f9;
        padding: 1rem;
        border-radius: 24px;
        box-shadow: 
            0 0 0 1px rgba(99,102,241,0.3),
            0 12px 48px rgba(0,0,0,0.8),
            0 0 80px rgba(99,102,241,0.25);
        border: 1px solid rgba(99,102,241,0.3);
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        width: 100%;
        position: relative;
        overflow: hidden;
    }

    .webcam-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at center, rgba(99,102,241,0.1) 0%, transparent 70%);
        pointer-events: none;
    }

    /* Webcam inner wrapper – invisible & only used for centering */
    .webcam-inner {
        padding: 0;
        border-radius: 0;
        background: transparent;
        border: none;
        box-shadow: none;
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Center the webcam preview image */
    .webcam-container img,
    .webcam-container [data-testid="stImage"] {
        border-radius: 16px;
        box-shadow:
            0 0 0 1px rgba(99,102,241,0.3),
            0 12px 40px rgba(0,0,0,0.8),
            0 0 60px rgba(99,102,241,0.3);
        border: 2px solid rgba(99,102,241,0.4);
        max-width: 480px;          /* control webcam box size */
        width: 100%;
        margin: 0 auto;
        display: block;
    }

    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #7c3aed 100%);
        color: #ffffff;
        border-radius: 999px;
        border: 1px solid rgba(139,92,246,0.5);
        padding: 0.85rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 
            0 0 20px rgba(99,102,241,0.4),
            0 4px 16px rgba(0,0,0,0.5),
            inset 0 1px 0 rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s ease;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 
            0 0 30px rgba(99,102,241,0.6),
            0 8px 24px rgba(0,0,0,0.6),
            inset 0 1px 0 rgba(255,255,255,0.3);
        border-color: rgba(139,92,246,0.8);
    }

    .stButton > button:active {
        transform: translateY(0px) scale(0.98);
    }

    /* Enhanced progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        box-shadow: 0 0 20px rgba(99,102,241,0.5);
        border-radius: 999px;
    }

    .stProgress > div > div > div {
        background: rgba(99,102,241,0.2);
        border-radius: 999px;
    }

     /* Emotion result card - Enhanced (compact for screen fit) */
     .result-card {
         padding: 1.8rem 2rem;
         border-radius: 20px;
         margin-top: 0.5rem;
         background: linear-gradient(135deg, rgba(30,27,75,0.95) 0%, rgba(15,23,42,0.95) 100%);
         border: 1px solid rgba(99,102,241,0.3);
         box-shadow: 
             0 0 0 1px rgba(99,102,241,0.2),
             0 12px 48px rgba(0,0,0,0.8),
             0 0 80px rgba(99,102,241,0.25);
         text-align: center;
         animation: slideUpFade 0.5s ease-out;
         position: relative;
         overflow: hidden;
         width: 100%;
     }

    /* Hide yellow circle overlay */
    .result-card::before {
        display: none;
    }

    @keyframes slideUpFade {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .result-emoji {
        font-size: 2rem;
        margin-bottom: 0.3rem;
        filter: drop-shadow(0 0 20px rgba(255,255,255,0.3));
        animation: float 3s ease-in-out infinite;
        position: relative;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .result-label {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        text-shadow: 0 0 30px currentColor;
        letter-spacing: -1px;
        position: relative;
    }

    .result-confidence {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-bottom: 0.2rem;
        font-weight: 500;
        position: relative;
    }

    .result-desc {
        font-size: 0.75rem;
        color: #9ca3af;
        position: relative;
        margin-top: 0.2rem;
    }

    /* Enhanced prediction list */
    .prediction-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.8rem 1rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(30,27,75,0.8) 0%, rgba(15,23,42,0.8) 100%);
        margin-bottom: 0.6rem;
        border: 1px solid rgba(99,102,241,0.2);
        transition: all 0.3s ease;
        position: relative;
    }

    .prediction-row:hover {
        transform: translateX(4px);
        border-color: rgba(99,102,241,0.4);
        box-shadow: 0 4px 16px rgba(99,102,241,0.2);
    }

    .prediction-emoji {
        font-size: 1.8rem;
        width: 2.5rem;
        text-align: center;
        filter: drop-shadow(0 0 8px currentColor);
    }

    .prediction-name {
        font-weight: 600;
        flex: 1;
        font-size: 1rem;
        color: #f1f5f9;
    }

    .prediction-score {
        font-weight: 700;
        font-size: 1rem;
    }

    .prediction-bar-bg {
        width: 100%;
        height: 8px;
        border-radius: 999px;
        background: rgba(55,65,81,0.9);
        overflow: hidden;
        margin-top: 0.4rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }

    .prediction-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.5s ease;
        box-shadow: 0 0 10px currentColor;
    }

     .info-box {
         margin-top: 1.5rem;
         padding: 1.8rem 2rem;
         border-radius: 16px;
         background: linear-gradient(135deg, rgba(30,27,75,0.9) 0%, rgba(15,23,42,0.9) 100%);
         border-left: 4px solid #6366f1;
         font-size: 1rem;
         color: #e0e7ff;
         box-shadow: 
             0 4px 20px rgba(0,0,0,0.6),
             0 0 40px rgba(99,102,241,0.2);
         backdrop-filter: blur(10px);
         width: 100%;
     }

    .info-box strong {
        font-weight: 700;
        color: #c7d2fe;
    }

    /* Enhanced emotion chips */
    .emotion-chip {
        text-align: center;
        background: linear-gradient(135deg, rgba(30,27,75,0.9) 0%, rgba(15,23,42,0.9) 100%);
        border-radius: 16px;
        padding: 1.2rem 0.8rem;
        border: 1px solid rgba(99,102,241,0.2);
        box-shadow: 
            0 0 0 1px rgba(99,102,241,0.1),
            0 4px 20px rgba(0,0,0,0.7),
            0 0 40px rgba(99,102,241,0.15);
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .emotion-chip:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 
            0 0 0 1px rgba(99,102,241,0.4),
            0 8px 32px rgba(0,0,0,0.8),
            0 0 60px rgba(99,102,241,0.3);
        border-color: rgba(99,102,241,0.4);
    }

    .emotion-chip-emoji {
        font-size: 2.4rem;
        margin-bottom: 0.4rem;
        filter: drop-shadow(0 0 10px rgba(255,255,255,0.2));
    }

    .emotion-chip-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f1f5f9;
    }

    /* Sidebar styling enhancements */
    section[data-testid="stSidebar"] .element-container {
        color: #e0e7ff;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(99,102,241,0.3);
    }

    section[data-testid="stSidebar"] strong {
        color: #c7d2fe;
    }

    section[data-testid="stSidebar"] p {
        color: #cbd5e1;
    }

    /* Info/warning/success boxes */
    .stAlert {
        background: linear-gradient(135deg, rgba(30,27,75,0.95) 0%, rgba(15,23,42,0.95) 100%);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }

    #MainMenu, footer {visibility: hidden;}

    /* Tab styling - increase font size (2rem) */
    [data-testid="stTabs"] [data-baseweb="tab"] span,
    [data-testid="stTabs"] button span {
        font-size: 2rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab"],
    [data-testid="stTabs"] button {
        padding: 0.75rem 1.5rem !important;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.3rem;
        }
        .hero-section {
            padding: 2rem 1.5rem;
        }
        .result-emoji {
            font-size: 3.5rem;
        }
        .result-label {
            font-size: 2rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Emotion configuration
# ─────────────────────────────────────────────────────────────
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

EMOTION_EMOJIS = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲",
}

EMOTION_COLORS = {
    "Angry": "#ef4444",
    "Disgust": "#8b4513",
    "Fear": "#a855f7",
    "Happy": "#facc15",
    "Neutral": "#9ca3af",
    "Sad": "#3b82f6",
    "Surprise": "#fb923c",
}

EMOTION_DESCRIPTIONS = {
    "Angry": "Facial cues suggest irritation, frustration, or anger.",
    "Disgust": "Expression indicates dislike or aversion to something.",
    "Fear": "Signals of anxiety or concern are visible in the face.",
    "Happy": "Smile and relaxed features show positive affect.",
    "Neutral": "No strong emotional expression; a calm resting face.",
    "Sad": "Drooping features suggest sadness or low mood.",
    "Surprise": "Raised brows and open eyes indicate surprise or shock.",
}

IMG_SIZE = 224

# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────
MODEL_PATH = None
for candidate in [
    "mod_my_model01.keras",
    "../mod_my_model01.keras",
    "../model/mod_my_model01.keras",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "mod_my_model01.keras"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "mod_my_model01.keras"),
]:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break

if MODEL_PATH is None:
    MODEL_PATH = "mod_my_model01.keras"  # fall back path


@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info(f"Tried model path: {MODEL_PATH}")
        return None


# ─────────────────────────────────────────────────────────────
# Image / prediction utilities
# ─────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image, img_size: int = IMG_SIZE) -> np.ndarray:
    """Resize and normalize a face image for model prediction."""
    arr = np.array(image)

    if len(arr.shape) == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

    arr_resized = cv2.resize(arr, (img_size, img_size))
    arr_norm = arr_resized.astype(np.float32) / 255.0
    arr_batch = np.expand_dims(arr_norm, axis=0)
    return arr_batch


def detect_face_pil(image: Image.Image):
    """Detect faces in a PIL image; return bounding boxes and underlying array."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    arr = np.array(image)

    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr

    faces = cascade.detectMultiScale(gray, 1.1, 4)
    return faces, arr


def predict_emotion(model, face_image: Image.Image):
    """Predict the dominant emotion for a cropped face."""
    batch = preprocess_image(face_image)
    preds = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(preds))
    main_emotion = EMOTION_LABELS[idx]
    confidence = float(preds[idx])
    all_predictions = {EMOTION_LABELS[i]: float(preds[i]) for i in range(len(EMOTION_LABELS))}
    return main_emotion, confidence, all_predictions


# ─────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────
def show_hero_section():
    st.markdown(
        '<div class="main-block"><div class="hero-section"><div class="hero-layout"><div class="hero-left"><div class="hero-pill"><span class="hero-pill-dot"></span>REAL-TIME DEEP LEARNING</div><div class="hero-title">AI Emotion Recognition</div><div class="hero-subtitle">Real-time facial emotion detection powered by a deep convolutional neural network. Position your face in front of the camera and watch the model infer your dominant emotion frame by frame.</div><div class="hero-badges"><span class="hero-badge"><span class="hero-badge-icon">🧠</span> TensorFlow · Keras</span><span class="hero-badge"><span class="hero-badge-icon">📷</span> OpenCV · Haar Cascade</span><span class="hero-badge"><span class="hero-badge-icon">🌐</span> Streamlit UI</span></div></div><div class="hero-right"><div class="hero-right-card"><div class="hero-right-label">Session overview</div><div class="hero-stat-grid"><div class="hero-stat"><span>Classes</span><span>7 emotions</span></div><div class="hero-stat"><span>Input size</span><span>224 × 224</span></div><div class="hero-stat"><span>Mode</span><span>Live webcam</span></div></div><div class="hero-caption">The model processes each frame, extracts facial features, and outputs a probability distribution across all emotion classes in real time.</div></div></div></div></div></div>',
        unsafe_allow_html=True,
    )


def show_prediction_result(emotion, confidence, all_predictions):
    color = EMOTION_COLORS.get(emotion, "#6366f1")
    emoji = EMOTION_EMOJIS.get(emotion, "😊")
    desc = EMOTION_DESCRIPTIONS.get(emotion, "")

    st.markdown(
        f"""
<div class="result-card">
  <div class="result-emoji">{emoji}</div>
  <div class="result-label" style="color:{color};">{emotion}</div>
  <div class="result-confidence">Confidence: {confidence:.1%}</div>
  <div class="result-desc">{desc}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.progress(confidence)

    st.markdown("#### Detailed probabilities")
    sorted_preds = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)

    for em, prob in sorted_preds:
        e_color = EMOTION_COLORS.get(em, "#6366f1")
        e_emoji = EMOTION_EMOJIS.get(em, "😊")
        st.markdown(
            f"""
<div class="prediction-row">
  <div class="prediction-emoji">{e_emoji}</div>
  <div style="flex:1;">
    <div class="prediction-name">{em}</div>
    <div class="prediction-bar-bg">
      <div class="prediction-bar-fill" style="width:{prob*100:.1f}%; background:{e_color};"></div>
    </div>
  </div>
  <div class="prediction-score" style="color:{e_color};">{prob:.1%}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="info-box">
  <strong>How the model works.</strong> The network processes your face image, 
  extracts high-level features (such as eye shape, mouth curvature, and brow position), 
  and estimates a probability for each emotion class. The displayed label corresponds to 
  the class with the highest probability.
</div>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────
def main():
    model = load_model()
    if model is None:
        st.stop()

    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False

    show_hero_section()

    # Sidebar
    st.sidebar.title("🔍 Model overview")
    st.sidebar.markdown(
        """
- Backbone: transfer-learning CNN  
- Input size: **224 × 224** RGB  
- Framework: **TensorFlow · Keras**  
- Vision stack: **OpenCV + Haar Cascade**  
- UI: **Streamlit**
"""
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "For robust predictions, keep a single face in frame, with good lighting and frontal pose."
    )

    # Tabs: live vs upload
    live_tab, upload_tab = st.tabs(["🎥 Live Detection", "📸 Image Upload"])

    # ── Live detection tab ────────────────────────────────────
    with live_tab:
        st.markdown(
            """
<div class="card">
  <div class="card-header">📷 Live real-time emotion detection</div>
  <p style="font-size:0.9rem;color:#9ca3af;margin-bottom:0.8rem;">
    Use your webcam to stream live video. The model will periodically analyze your face and display 
    the dominant emotion with its confidence level.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )

        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶️ Start live detection", use_container_width=True):
                st.session_state.webcam_active = True
                st.rerun()
        with col_stop:
            if st.button("⏹ Stop", use_container_width=True):
                st.session_state.webcam_active = False
                st.rerun()

        if st.session_state.webcam_active:
          
            # Center the webcam in the middle column
            left, middle, right = st.columns([1, 2, 1])
            with middle:
                video_placeholder = st.empty()   # no extra HTML wrapper

            emotion_placeholder = st.empty()
            confidence_placeholder = st.empty()

            # OpenCV webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Unable to access webcam. Check camera permissions.")
                st.session_state.webcam_active = False
            else:
                cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                last_emotion = None
                last_conf = 0.0
                frame_count = 0

                while st.session_state.webcam_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from webcam.")
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(gray, 1.1, 4)

                    if len(faces) > 0 and frame_count % 5 == 0:
                        x, y, w, h = faces[0]
                        face_roi = frame_rgb[y : y + h, x : x + w]
                        try:
                            emotion, conf, all_preds = predict_emotion(
                                model, Image.fromarray(face_roi)
                            )
                            last_emotion = emotion
                            last_conf = conf
                        except Exception:
                            pass

                    # draw overlay
                    for (x, y, w, h) in faces:
                        color_hex = EMOTION_COLORS.get(last_emotion, "#6366f1")
                        color_rgb = tuple(int(color_hex[i : i + 2], 16) for i in (1, 3, 5))
                        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), color_rgb, 2)
                        if last_emotion:
                            label = f"{EMOTION_EMOJIS.get(last_emotion, '😊')} {last_emotion} ({last_conf:.0%})"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            (tw, th), baseline = cv2.getTextSize(label, font, 0.6, 2)
                            cv2.rectangle(
                                frame_rgb,
                                (x, y - th - 10),
                                (x + tw, y),
                                color_rgb,
                                -1,
                            )
                            cv2.putText(
                                frame_rgb,
                                label,
                                (x, y - 5),
                                font,
                                0.6,
                                (255, 255, 255),
                                2,
                            )

                    # show frame in center column - let CSS + max-width control the size
                    with middle:
                        video_placeholder.image(
                            frame_rgb,
                            channels="RGB",
                            use_container_width=True,  # let CSS + max-width control the size
                        )

                    if last_emotion:
                        color = EMOTION_COLORS.get(last_emotion, "#6366f1")
                        emoji = EMOTION_EMOJIS.get(last_emotion, "😊")
                        desc = EMOTION_DESCRIPTIONS.get(last_emotion, "")
                        emotion_placeholder.markdown(
                            f"""
                                <div class="result-card">
                                <div class="result-emoji">{emoji}</div>
                                <div class="result-label" style="color:{color};">{last_emotion}</div>
                                <div class="result-confidence">Confidence: {last_conf:.1%}</div>
                                <div class="result-desc">{desc}</div>
                                </div>
                                """,
                            unsafe_allow_html=True,
                        )
                        confidence_placeholder.progress(last_conf)
                    else:
                        emotion_placeholder.info("Align your face with the camera.")
                        confidence_placeholder.empty()

                    frame_count += 1
                    time.sleep(0.03)

                cap.release()
        else:
            st.info("Press **Start live detection** to activate the webcam.")

    # ── Upload tab ────────────────────────────────────────────
    with upload_tab:
        st.markdown(
            """
<div class="card">
  <div class="card-header">📸 Single-image emotion analysis</div>
  <p style="font-size:0.9rem;color:#9ca3af;margin-bottom:0.8rem;">
    Upload a face image. The system will detect the face and classify the dominant emotion.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )

        file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if file is not None:
            image = Image.open(file).convert("RGB")

            faces, arr = detect_face_pil(image)
            if len(faces) == 0:
                st.error("No face detected. Try another image with a clear frontal face.")
            else:
                # show faces overlay
                img_draw = arr.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(img_draw, (x, y), (x + w, y + h), (99, 102, 241), 2)

                # Horizontal layout - side by side images (minimal gap)
                spacer, col_left, col_right, spacer1 = st.columns([1, 1, 1, 1]) 
                
                with col_left:
                    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
                    st.markdown("**Input image**")
                    st.image(image, width=350, use_container_width=False)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_right:
                    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
                    st.markdown("**Detected face(s)**")
                    st.image(img_draw, width=350, use_container_width=False)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Center the analyze button
                col_btn_left, col_btn_center, col_btn_right = st.columns([1, 2, 1])
                with col_btn_center:
                    if st.button("🔮 Analyze emotion", use_container_width=True):
                        x, y, w, h = faces[0]
                        face_roi = image.crop((x, y, x + w, y + h))
                        emotion, conf, all_preds = predict_emotion(model, face_roi)
                        show_prediction_result(emotion, conf, all_preds)

    # ── Reference chips ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📚 Emotion categories")

    cols = st.columns(len(EMOTION_LABELS))
    for em, col in zip(EMOTION_LABELS, cols):
        with col:
            st.markdown(
                f"""
<div class="emotion-chip">
  <div class="emotion-chip-emoji">{EMOTION_EMOJIS[em]}</div>
  <div class="emotion-chip-label">{em}</div>
</div>
""",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
