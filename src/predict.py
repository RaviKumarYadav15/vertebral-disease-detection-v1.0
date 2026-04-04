import tensorflow as tf
import cv2
import numpy as np
import os
from src.config import MODELS_DIR
from src.preprocess import process_image


# --- 1. LOAD BOTH MODELS ---
SPINE_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_spine_v1.keras")
GATEKEEPER_MODEL_PATH = os.path.join(MODELS_DIR, "gatekeeper_v1.keras")

try:
    spine_model = tf.keras.models.load_model(SPINE_MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading Specialist model: {e}")
    spine_model = None

try:
    gatekeeper_model = tf.keras.models.load_model(GATEKEEPER_MODEL_PATH)
    print("🛡️ Gatekeeper Security is ONLINE.")
except Exception as e:
    print("⚠️ Gatekeeper Model not found. Please run src.train_gatekeeper first.")
    gatekeeper_model = None

# simple validation (failed code)
# def is_valid_xray(img_array):
#     """
#     An advanced 'bouncer' that ignores JPEG noise and looks at structural color distribution.
#     """
#     # 1. Solid Background Check
#     # Find the most common pixel color in the image
#     unique, counts = np.unique(img_array, return_counts=True)
#     max_color_ratio = np.max(counts) / img_array.size
    
#     # If one single exact color makes up more than 50% of the image, 
#     # it is almost certainly a digital diagram, not an organic X-ray.
#     if max_color_ratio > 0.50:
#         return False
        
#     # 2. Quantized Color Count (Stripping JPEG Noise)
#     # We group similar colors together (e.g., rounding down to the nearest 10)
#     quantized_img = (img_array // 10) * 10
#     unique_colors = len(np.unique(quantized_img))
    
#     # After stripping the noise, organic X-rays still have dozens of gray zones.
#     # Diagrams will collapse down to just a few core colors.
#     if unique_colors < 15:
#         return False
        
#     return True

def predict_xray(image_path):
    """
    Two-Stage Enterprise Inference Pipeline.
    """
    if spine_model is None:
        return "Error: Specialist Model not found", 0.0, None

    # Process Image
    processed_img = process_image(image_path)
    if processed_img is None:
        return "Error: Could not read image", 0.0, None

    # Prepare Tensor (1, 224, 224, 1)
    input_tensor = np.expand_dims(processed_img, axis=0)
    input_tensor = np.expand_dims(input_tensor, axis=-1)

    # --- STAGE 1: THE GATEKEEPER CHECK ---
    if gatekeeper_model is not None:
        is_xray_prob = gatekeeper_model.predict(input_tensor, verbose=0)[0][0]
        
        # If it's closer to 0, it's a random image/pie chart
        if is_xray_prob < 0.5:
            fake_confidence = (1.0 - is_xray_prob) * 100
            return f"Access Denied: AI Bouncer rejected image ({fake_confidence:.1f}% sure it is not an X-ray).", 0.0, processed_img

    # --- STAGE 2: THE SPECIALIST DIAGNOSIS ---
    # The image passed the check, so the Specialist looks for disease.
    prediction = spine_model.predict(input_tensor, verbose=0)[0][0]

    if prediction < 0.5:
        diagnosis = "Abnormal / Diseased"
        confidence = (1.0 - prediction) * 100 
    else:
        diagnosis = "Healthy"
        confidence = prediction * 100

    return diagnosis, confidence, processed_img