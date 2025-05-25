# ocr_utils.py
import pytesseract
from PIL import Image

def extract_text_from_image(pil_image):
    # Use Tesseract OCR to extract text from preprocessed PIL Image
    text = pytesseract.image_to_string(pil_image, lang='eng')
    return text
