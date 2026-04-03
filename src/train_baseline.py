import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib # Used to save the non-deep learning model
from src.config import DATA_PROCESSED_DIR, MODELS_DIR, IMAGE_WIDTH, IMAGE_HEIGHT

def load_data_flat(data_dir):
    """Loads images and flattens them into 1D arrays for the baseline model."""
    X = []
    y = []
    
    # TensorFlow alphabetical mapping: abnormal=0, healthy=1
    categories = ['abnormal', 'healthy']
    
    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Flatten the (224, 224) image into a (50176,) array
                X.append(img.flatten() / 255.0) 
                y.append(label)
                
    return np.array(X), np.array(y)

def train_baseline():
    print("📦 Loading and flattening data for Baseline...")
    X, y = load_data_flat(DATA_PROCESSED_DIR)
    
    # Simple 80/20 split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Training Logistic Regression Baseline (this might take a minute)...")
    # We increase max_iter so the math has time to converge
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- Baseline Results ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Abnormal', 'Healthy']))

    # Save the baseline model
    save_path = os.path.join(MODELS_DIR, "baseline_logistic.pkl")
    joblib.dump(model, save_path)
    print(f"✅ Baseline model saved to {save_path}")

if __name__ == "__main__":
    train_baseline()