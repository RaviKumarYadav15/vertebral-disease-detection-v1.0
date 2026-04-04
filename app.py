import streamlit as st
import os
from src.predict import predict_xray
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Spine Fracture AI", 
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
# LEFT COLUMN: Title, Dropdowns, Uploader, and Results
# ==========================================
with left_col:
    st.title("⚕️ Spine Fracture Detection")
    st.markdown("Upload an Axial Cervical CT slice to instantly check for neck fractures using our custom Deep Learning model.")
    
    # --- CLINICAL CONTEXT SECTION (DROPDOWNS) ---
    st.markdown("### 🔬 Analyzing Cervical Spine CT Scans")
    
    with st.expander("🦴 What is the Cervical Spine?"):
        st.markdown("<p style='font-size: 1.1em;'>In medicine, <b>'cervical'</b> refers to the neck. The cervical spine consists of the top 7 vertebrae (C1-C7) that connect the skull to the shoulders.</p>", unsafe_allow_html=True)
        
    with st.expander("📷 What is an Axial CT Scan?"):
        st.markdown("<p style='font-size: 1.1em;'>Unlike a standard 2D side-profile X-ray, an <b>Axial CT scan</b> takes horizontal \"slices\" of the body. You are looking straight down through the neck. The solid white ring in these scans is the bone protecting the spinal cord.</p>", unsafe_allow_html=True)
        
    with st.expander("🚨 The Clinical Problem"):
        st.markdown("<p style='font-size: 1.1em;'>A cervical fracture (broken neck) is a critical medical emergency. Detecting micro-fractures in these complex 3D cross-sections is incredibly difficult but vital to prevent severe nerve damage. This AI acts as an automated, high-speed screening tool.</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- UPLOADER & DIAGNOSIS (SIDE-BY-SIDE) ---
    upload_col, diag_col = st.columns(2, gap="medium")

    # Uploader goes in the left sub-column
    with upload_col:
        uploaded_file = st.file_uploader("Choose an Axial CT Slice (JPG/PNG)", type=["jpg", "jpeg", "png"])

    # Define variables so they exist outside the if-statement
    is_invalid_image = False 
    diagnosis = ""
    confidence = 0.0
    processed_img = None

    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with upload_col: # Keep the loading spinner under the uploader
            with st.spinner("🤖 AI is analyzing..."):
                diagnosis, confidence, processed_img = predict_xray(file_path)
            
        # Determine if the diagnosis is an error from the Gatekeeper
        is_invalid_image = "Error" in diagnosis or "Oops" in diagnosis or "Access Denied" in diagnosis
        os.remove(file_path)

        # Diagnosis Results go in the right sub-column
        with diag_col:
            st.subheader("📊 Diagnosis")
            if is_invalid_image:
                st.error(f"**{diagnosis}** 🛑")
            elif "Abnormal" in diagnosis or "Fracture" in diagnosis or "Disease" in diagnosis:
                st.error("**Fracture Detected** 🚨")
                st.warning(f"**Confidence:** {confidence:.2f}%")
                st.write("*Patterns consistent with a cervical fracture.*")
            else:
                st.success("**Normal (No Fracture)** ✅")
                st.info(f"**Confidence:** {confidence:.2f}%")
                st.write("*No evidence of fracture detected.*")
    
    # If no file is uploaded yet, prompt the user in the diagnosis column
    else:
        with diag_col:
            st.markdown("<br><br>", unsafe_allow_html=True) # Adds a little vertical space
            st.info("👈 Upload an image to see results here.")


# ==========================================
# RIGHT COLUMN: Image Viewer
# ==========================================
with right_col:
    if uploaded_file is not None:
        
        # --- IF BLOCKED: Show rejected image ---
        if is_invalid_image:
            st.header("👁️ Uploaded Image")
            st.markdown("<br>", unsafe_allow_html=True)
            
            display_img = Image.open(uploaded_file)
            display_img.thumbnail((600, 600)) 
            st.image(display_img, caption="Rejected Input: Not a Valid Scan")
            
        # --- IF VALID: Show AI vision ---
        else:
            st.header("👁️ AI Vision Analysis")
            st.markdown("<br>", unsafe_allow_html=True)
            
            img1, img2 = st.columns(2)
            with img1:
                st.image(uploaded_file, caption="Original Upload", use_column_width=True)
            with img2:
                if processed_img is not None:
                    st.image(processed_img, caption="AI Vision [CLAHE] Enhanced", use_column_width=True, clamp=True, channels="GRAY")

# ==========================================
# FOOTER: Disclaimer & Copyright
# ==========================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.9em;'>", unsafe_allow_html=True)
st.write("**Disclaimer:** This is an AI proof-of-concept built for educational purposes. It should not be used for actual medical diagnosis.")
st.write("**© copyrights . all rights reserved ravi@official2026**")
st.markdown("</div>", unsafe_allow_html=True)