import os
import shutil
import pytesseract
import pandas as pd


# Find Tesseract automatically
tesseract_path = shutil.which("tesseract")

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


def extract_text_with_confidence(image, psm=3):
    """
    Extract text and calculate the average OCR confidence.
    """

    config = f"--psm {psm}"

    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=pytesseract.Output.DATAFRAME
    )

    # Remove empty OCR results
    data = data.dropna(subset=["text"])

    data = data[data["text"].str.strip() != ""]

    # Keep only valid confidence values
    confidence_values = data["conf"]

    confidence_values = confidence_values[
        confidence_values >= 0
    ]

    if len(confidence_values) > 0:
        average_confidence = confidence_values.mean()
    else:
        average_confidence = 0.0

    text = " ".join(
        data["text"].astype(str).tolist()
    ).strip()

    return text, float(average_confidence)