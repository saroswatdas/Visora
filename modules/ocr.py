import os
import shutil
import pytesseract


# Find Tesseract automatically
tesseract_path = shutil.which("tesseract")

# Windows fallback
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
elif os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


def extract_text(image, psm=3):
    """
    Extract text from an image using Tesseract OCR.
    """

    config = f"--psm {psm}"

    text = pytesseract.image_to_string(
        image,
        config=config
    )

    return text.strip()