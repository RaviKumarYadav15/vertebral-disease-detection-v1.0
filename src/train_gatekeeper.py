# src/train_gatekeeper.py
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from src.config import MODELS_DIR

# --- 1. CONFIGURATION ---
# Point this to your new folder where you placed the X-rays and Random images
DATA_DIR = os.path.join("data", "gatekeeper_data")
BATCH_SIZE = 16
IMG_SIZE = (224, 224)

print("🛡️ Booting up Gatekeeper Training Pipeline...")

# --- 2. LOAD DATASET ---
# 0 = random, 1 = xrays (assuming alphabetical folder names)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale" # We force everything to grayscale to match the pipeline
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

# --- 3. BUILD THE FAST GATEKEEPER MODEL ---
# This is a very lightweight model designed for speed.
gatekeeper = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 1)),
    
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid') # 0 = Random, 1 = Xray
])

gatekeeper.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- 4. TRAIN & SAVE ---
print("\n🚀 Training the Gatekeeper...")
# It only needs a few epochs because this is an easy visual task
gatekeeper.fit(train_dataset, validation_data=val_dataset, epochs=5)

SAVE_PATH = os.path.join(MODELS_DIR, "gatekeeper_v1.keras")
gatekeeper.save(SAVE_PATH)
print(f"\n✅ Gatekeeper Model saved successfully at: {SAVE_PATH}")