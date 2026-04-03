import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- IMAGE PROCESSING ---
# 224x224 is the standard for most CNNs, including ResNet later on
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CHANNELS = 1 # 1 for Grayscale, 3 for RGB

# --- DEEP LEARNING HYPERPARAMETERS ---
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001