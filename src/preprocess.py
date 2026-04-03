import cv2
import os
import glob
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, IMAGE_WIDTH, IMAGE_HEIGHT

def process_image(image_path):
    """
    Reads an image, applies grayscale, CLAHE, Gaussian Blur, and resizes.
    """
    # 1. Read image in grayscale explicitly
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This enhances local contrast to make vertebrae edges visible
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)

    # 3. Apply Gaussian Blur (Noise Reduction)
    # Removes high-frequency noise while preserving bone structure
    blurred_img = cv2.GaussianBlur(enhanced_img, (3, 3), 0)

    # 4. Resize to the target dimensions defined in config (224x224)
    resized_img = cv2.resize(blurred_img, (IMAGE_WIDTH, IMAGE_HEIGHT))

    return resized_img

def process_directory(category_name):
    """
    Processes all images in a specific category folder (e.g., 'healthy').
    """
    input_dir = os.path.join(DATA_RAW_DIR, category_name)
    output_dir = os.path.join(DATA_PROCESSED_DIR, category_name)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    
    # Robust file finding: only look for specific image extensions
    image_paths = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        found = glob.glob(os.path.join(input_dir, ext))
        image_paths.update(found) # .update() on a set ignores duplicates
    
    image_paths = list(image_paths) # Convert back to list for the loop
    print(f"Found {len(image_paths)} unique images in '{category_name}'...")
    # image_paths = []
    # for ext in ['*.jpg', '*.jpeg', '*.png']:
    #     image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    #     # Also look for uppercase extensions just in case
    #     image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    # print(f"Found {len(image_paths)} valid images in '{category_name}'...")

    count = 0
    for path in image_paths:
        processed = process_image(path)
        if processed is not None:
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, processed)
            count += 1
            
    print(f"Successfully processed {count} images in '{category_name}'.")

if __name__ == "__main__":
    print("\n" + "="*40)
    print("⚕️  IMAGE PREPROCESSING PIPELINE STARTING")
    print("="*40)
    
    # Process both categories as per the alphabetical mapping requirement
    process_directory("abnormal")
    process_directory("healthy")
    
    print("\n✅ Pipeline Complete. Check your 'data/processed/' folder.")