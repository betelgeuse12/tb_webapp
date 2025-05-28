import cv2
import numpy as np

def preprocess_xray_image(image_path, size=299):
    # Load in grayscale for CLAHE
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError("Cannot read uploaded image for preprocessing.")

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)

    # Convert CLAHE output to 3 channels (for model)
    img_clahe_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

    # Resize
    img_resized = cv2.resize(img_clahe_rgb, (size, size))

    # Normalize to [0,1]
    img_normalized = img_resized / 255.0

    # Model expects shape (1, size, size, 3)
    img_ready = np.expand_dims(img_normalized, axis=0)
    return img_ready, img_resized  # Return both (for Grad-CAM overlay if needed)
