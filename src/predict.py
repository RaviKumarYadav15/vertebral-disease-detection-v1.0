import tensorflow as tf
import cv2
import numpy as np
from src.config import MODELS_DIR, IMAGE_WIDTH, IMAGE_HEIGHT
from src.preprocess import process_image
import os

# Load the model once when the script starts
MODEL_PATH = os.path.join(MODELS_DIR, "cnn_spine_v1.keras")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}. Make sure you have trained it first!")
    model = None

def predict_xray(image_path):
    """
    Takes an X-ray image path, processes it, and returns the diagnosis.
    """
    if model is None:
        return "Model not found", 0.0, None

    # 1. Run the exact same OpenCV pipeline used for training
    processed_img = process_image(image_path)
    
    if processed_img is None:
        return "Invalid Image", 0.0, None

    # 2. Prepare the image for the Neural Network
    # Neural networks expect a "batch" of images, even if it's just one.
    # We change the shape from (224, 224) to (1, 224, 224, 1)
    input_tensor = np.expand_dims(processed_img, axis=0) # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=-1) # Add channel dimension (grayscale)

    # 3. Get the prediction
    # The output is a probability between 0.0 and 1.0
    prediction = model.predict(input_tensor)[0][0]

    # 4. Interpret the result based on ['abnormal', 'healthy'] mapping
    if prediction < 0.5:
        diagnosis = "Abnormal / Diseased"
        # Since 0 is the target for Abnormal, a 0.02 prediction 
        # means the model is 98% sure it's abnormal.
        confidence = (1.0 - prediction) * 100 
    else:
        diagnosis = "Healthy"
        # Since 1 is the target for Healthy, a 0.98 prediction
        # means the model is 98% sure it's healthy.
        confidence = prediction * 100

    # --- THE FIX: Return the processed_img so Streamlit can display it! ---
    return diagnosis, confidence, processed_img