# preprocessing.py
import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    # Load image with OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

    # Denoise image
    img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)

    # Binarize image using adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 2)

    # Convert back to PIL Image
    pil_img = Image.fromarray(img)
    return pil_img
