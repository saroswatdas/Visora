import cv2


def preprocess_image(img_bgr, mode="clean"):
    """
    Prepare an image for Tesseract.

    mode:
      "clean"    - grayscale + resize only (best for clean screenshots/scans)
      "otsu"     - Otsu binarization (good for high-contrast documents)
      "adaptive" - adaptive threshold (uneven lighting, photos of paper)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Keep width in a range Tesseract likes (about 1400-2400 px)
    h, w = gray.shape
    if w < 1400:
        scale = 1400 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    elif w > 2400:
        scale = 2400 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    if mode == "clean":
        return gray

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if mode == "otsu":
        _, out = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return out

    if mode == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 15,
        )

    return gray