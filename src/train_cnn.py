import tensorflow as tf
import os
import matplotlib.pyplot as plt
from src.config import DATA_PROCESSED_DIR, MODELS_DIR, BATCH_SIZE, EPOCHS, IMAGE_WIDTH, IMAGE_HEIGHT
from src.model_cnn import build_custom_cnn
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_model():
    print("Loading preprocessed images...")
    
    # 1. Load Data from Directories
    # This automatically labels your images based on the folder name (healthy=0, abnormal=1)
    # It also splits the data: 80% for training, 20% for testing/validation
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_PROCESSED_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        color_mode="grayscale",
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle=True # CRITICAL: Prevents the model from seeing all abnormal then all healthy
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_PROCESSED_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode="grayscale",
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle = True
    )

    # CHECKPOINT: Print the mapping to be 100% sure
    print(f"DEBUG: Class Names detected: {train_dataset.class_names}")
    # It should say ['abnormal', 'healthy'] -> 0 is abnormal, 1 is healthy.

    # 2. Performance Optimization
    # Keeps images in memory so the GPU doesn't have to wait for the hard drive
    AUTOTUNE = tf.data.AUTOTUNE
    # train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    # validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # Use only prefetch
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    # 3. Build and Compile the Model
    print("Building the CNN...")
    model = build_custom_cnn()
    
    # Using a slightly slower learning rate helps the model "catch" the patterns
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(
        optimizer=optimizer, # The standard, highly efficient learning algorithm
        loss='binary_crossentropy', # The standard loss function for Binary Classification
        metrics=['accuracy']
    )
    # --- NEW: Add the Early Stopping Referee ---
    early_stopper = EarlyStopping(
        monitor='val_loss',        # Watch the validation loss
        patience=3,                # If it doesn't improve for 3 epochs, stop!
        restore_best_weights=True  # Automatically roll back to the best weights
    )

    # NEW: If learning gets stuck, lower the speed automatically
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

    # Calculate weights to handle the slight imbalance
    total = 96 + 104
    weight_for_0 = (1 / 96) * (total / 2.0) # Abnormal
    weight_for_1 = (1 / 104) * (total / 2.0) # Healthy

    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Applying Class Weights: {class_weight}")

    # 4. Train the Model!
    print(f"Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[early_stopper,reduce_lr]
    )

    # 5. Save the trained weights
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "cnn_spine_v1.keras")
    model.save(model_path)
    print(f"Training complete! Model saved to {model_path}")

    # Plot results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend(); plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title("Loss")
    plt.savefig('training_history.png')

if __name__ == "__main__":
    train_model()