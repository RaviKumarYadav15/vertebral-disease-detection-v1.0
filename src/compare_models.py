import joblib
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from src.train_baseline import load_data_flat
from src.config import DATA_PROCESSED_DIR, MODELS_DIR

def compare():
    print("📂 Loading data for comparison...")
    X_flat, y_true = load_data_flat(DATA_PROCESSED_DIR)
    
    # 1. SPLIT DATA
    _, X_test_flat, _, y_test = train_test_split(
        X_flat, y_true, test_size=0.2, random_state=42
    )
    
    # 2. Load Baseline 
    baseline = joblib.load(os.path.join(MODELS_DIR, "baseline_logistic.pkl"))
    y_pred_base = baseline.predict(X_test_flat)
    
    # 3. Load CNN
    cnn = tf.keras.models.load_model(os.path.join(MODELS_DIR, "cnn_spine_v1.keras"))
    X_test_cnn = X_test_flat.reshape(-1, 224, 224, 1)

    # --- THE MAGIC FIX: PREVENT DOUBLE SCALING ---
    # If the max value is 1.0, it means load_data_flat already divided by 255.
    # Because our CNN has a Rescaling layer built-in, we must multiply by 255 to undo it!
    if np.max(X_test_cnn) <= 1.0:
        print("⚠️ Fix Applied: Reverting double-scaled pixels so the CNN can 'see'...")
        X_test_cnn = X_test_cnn * 255.0

    print("🧠 CNN evaluating unseen images...")
    predictions = cnn.predict(X_test_cnn)
    y_pred_cnn = (predictions > 0.5).astype(int).flatten()

    # 4. Print Comparison Table
    print("\n" + "="*45)
    print("   🏥 VERTEBRAL MODEL EVALUATION REPORT")
    print("   (Testing on Unseen Validation Data)")
    print("="*45)
    print(f"{'Metric':<15} | {'Baseline':<12} | {'CNN':<12}")
    print("-" * 45)
    
    metrics = [
        ("Accuracy", accuracy_score),
        ("Recall (Sens.)", recall_score),
        ("F1-Score", f1_score)
    ]
    
    for metric_name, func in metrics:
        m_base = func(y_test, y_pred_base)
        m_cnn = func(y_test, y_pred_cnn)
        print(f"{metric_name:<15} | {m_base:>11.2%} | {m_cnn:>11.2%}")
    print("="*45)

    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_cnn)
    print(f"\nCNN Confusion Matrix (Unseen Data):")
    print(f"True Abnormal: {cm[0][0]} | False Healthy: {cm[0][1]}")
    print(f"False Abnormal: {cm[1][0]} | True Healthy: {cm[1][1]}")

if __name__ == "__main__":
    compare()