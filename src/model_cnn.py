import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS

def build_custom_cnn():
    model = models.Sequential()
    
    # 1. Input Layer
    model.add(layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
    model.add(layers.Rescaling(1./255))

    # --- NEW: THE OVERFITTING SHIELD (Data Augmentation) ---
    # This randomly alters the images in memory so the model can't memorize them
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1)) # Rotate by 10%
    model.add(layers.RandomZoom(0.1))     # Zoom in/out by 10%

    # --- Feature Extraction Block 1 ---
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Feature Extraction Block 2 ---
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Feature Extraction Block 3 ---
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Classification Block ---
    # GlobalAveragePooling prevents the 12-Million parameter collapse
    model.add(layers.GlobalAveragePooling2D())
    
    model.add(layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
    # High Dropout is crucial for small datasets
    model.add(layers.Dropout(0.6)) 

    # Output Layer
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

if __name__ == "__main__":
    model = build_custom_cnn()
    model.summary()