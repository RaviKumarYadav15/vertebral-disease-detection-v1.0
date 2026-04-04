import streamlit as st
import os
from src.predict import predict_xray
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Vertebral Disease AI", 
    page_icon="⚕️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Create temporary upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- MASTER LAYOUT SPLIT ---
left_col, right_col = st.columns([1, 1.2], gap="large")

# ==========================================
# LEFT COLUMN: Title, Uploader, and Results
# ==========================================
with left_col:
    st.title("⚕️ Vertebral Disease Detection")
    st.markdown("Upload a spinal X-ray image to instantly check for structural abnormalities using our custom Deep Learning model.")
    
    uploaded_file = st.file_uploader("Choose a Spinal X-Ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

    # Define a variable to track if the Gatekeeper blocked the image
    is_invalid_image = False 

    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("🤖 AI is analyzing the spinal structure..."):
            diagnosis, confidence, processed_img = predict_xray(file_path)

        st.markdown("---")
        st.subheader("📊 Diagnosis Results")
        
        # Determine if the diagnosis is an error
        is_invalid_image = "Error" in diagnosis or "Oops" in diagnosis or "Access Denied" in diagnosis
        
        if is_invalid_image:
            st.error(f"**{diagnosis}** 🛑")
        elif "Abnormal" in diagnosis:
            st.error(f"**Diagnosis:** {diagnosis} 🚨")
            st.warning(f"**Confidence:** {confidence:.2f}%")
            st.write("*The neural network detected structural patterns consistent with vertebral disease. Please consult a radiologist.*")
        else:
            st.success(f"**Diagnosis:** {diagnosis} ✅")
            st.info(f"**Confidence:** {confidence:.2f}%")
            st.write("*The neural network did not detect major structural abnormalities.*")

        os.remove(file_path)
    
    else:
        st.info("Please upload an image file to begin the analysis.")


# ==========================================
# RIGHT COLUMN: Image Viewer
# ==========================================
with right_col:
    if uploaded_file is not None:
        
        # --- IF BLOCKED: Resize the invalid image so it doesn't break the UI ---
        if is_invalid_image:
            st.header("👁️ Uploaded Image")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Open and shrink ONLY the rejected image
            display_img = Image.open(uploaded_file)
            display_img.thumbnail((600, 600)) # Caps height and width at 600px
            
            # Display the shrunken image
            st.image(display_img, caption="Rejected Input: Not a Valid Scan")
            
        # --- IF VALID: Show the exact, untouched medical images ---
        else:
            st.header("👁️ AI Vision Analysis")
            st.markdown("<br>", unsafe_allow_html=True)
            
            img1, img2 = st.columns(2)
            
            with img1:
                # Use the raw, untouched uploaded_file
                st.image(uploaded_file, caption="Original Upload", use_column_width=True)
                
            with img2:
                if processed_img is not None:
                    # Use the raw, untouched processed image
                    st.image(processed_img, caption="AI Vision [CLAHE] Histogram Equalisation Enhanced)", use_column_width=True, clamp=True, channels="GRAY")

# ==========================================
# FOOTER: Disclaimer & Copyright
# ==========================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.9em;'>", unsafe_allow_html=True)
st.write("**Disclaimer:** This is an AI proof-of-concept built for educational purposes. It should not be used for actual medical diagnosis.")
st.write("**© copyrights . all rights reserved ravi@official2026**")
st.markdown("</div>", unsafe_allow_html=True)