# app/app.py
import streamlit as st
import os
import sys

# Tell Python to look in the root folder for imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from src.predict import predict_xray
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Vertebral Disease AI", 
    page_icon="⚕️",
    layout="centered"
)

st.title("⚕️ Vertebral Disease Detection System")
st.write("Upload a spinal X-ray image to instantly check for structural abnormalities using our custom Deep Learning model.")

# Create temporary upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose a Spinal X-Ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Save file temporarily
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.markdown("---")
    
    # 2. UI Layout for Images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🩻 Original X-Ray")
        st.markdown("<br>", unsafe_allow_html=True)  # spacer
        st.image(uploaded_file, use_column_width=True)


    with st.spinner("🤖 AI is analyzing the spinal structure..."):
        # 3. Run Inference from predict.py
        diagnosis, confidence, processed_img = predict_xray(file_path)

    with col2:
        st.subheader("🧠 AI Vision (CLAHE Enhanced)")
        if processed_img is not None:
            st.image(processed_img, use_column_width=True, clamp=True)

    # 4. Display Results beautifully
    st.markdown("### 📊 Diagnosis Results")
    if "Error" in diagnosis:
        st.error(diagnosis)
    elif "Abnormal" in diagnosis:
        st.error(f"**Diagnosis:** {diagnosis} 🚨")
        st.warning(f"**Confidence:** {confidence:.2f}%")
        st.write("*The neural network detected structural patterns consistent with vertebral disease.*** Please consult a radiologist.*")
    else:
        st.success(f"**Diagnosis:** {diagnosis} ✅")
        st.info(f"**Confidence:** {confidence:.2f}%")
        st.write("*The neural network did not detect major structural abnormalities.*")

    # 5. Cleanup temp file
    os.remove(file_path)

st.markdown("---")
st.caption("Disclaimer: This is an AI proof-of-concept built for educational purposes. It should not be used for actual medical diagnosis.")