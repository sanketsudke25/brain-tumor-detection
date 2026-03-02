import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_trained_model():
    return load_model("brain_tumor_cnn_model.h5")

model = load_trained_model()

# ------------------ PREDICTION FUNCTION ------------------
def predict_brain_tumor(img):

    img = img.convert("RGB")          # force 3 channels
    img = img.resize((64,64))         # same as training
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    label = "Tumor Detected" if pred > 0.75 else "No Tumor Detected"
    probability = pred*100 if pred > 0.75 else (1-pred)*100

    return label, probability

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
}

/* Main Card */
.main-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(15px);
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    text-align: center;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #dcdcdc;
    margin-bottom: 30px;
}

/* Upload Box */
section[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 15px;
}

/* Prediction Box */
.result-box {
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    font-size: 22px;
    font-weight: bold;
}

/* Success Style */
.success {
    background-color: rgba(0, 255, 127, 0.15);
    color: #00ff7f;
    border: 1px solid #00ff7f;
}

/* Danger Style */
.danger {
    background-color: rgba(255, 0, 0, 0.15);
    color: #ff4d4d;
    border: 1px solid #ff4d4d;
}

/* Confidence */
.confidence {
    font-size: 18px;
    color: #ffffff;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ------------------ MAIN CARD ------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an MRI image to analyze tumor presence using AI</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", width=350)

    with st.spinner("🔍 Analyzing MRI... Please wait"):
        result, prob = predict_brain_tumor(image)

    if result == "Tumor Detected":
        st.markdown(f'<div class="result-box danger">⚠ {result}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box success">✅ {result}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="confidence">Confidence Level: {prob:.2f}%</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
