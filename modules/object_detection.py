from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

# Project root = the folder that contains app.py and modules/
ROOT = Path(__file__).resolve().parent.parent

# Folders searched (in order) for the model files
SEARCH_DIRS = [ROOT / "models", ROOT / "model", ROOT]

# Pascal VOC classes (the 20 classes this model knows)
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]


def _find_file(patterns):
    """Return the first file in SEARCH_DIRS matching any glob pattern."""
    for folder in SEARCH_DIRS:
        if not folder.is_dir():
            continue
        for pattern in patterns:
            matches = sorted(folder.glob(pattern))
            if matches:
                return matches[0]
    return None


@lru_cache(maxsize=1)
def _load_net():
    prototxt = _find_file(["*.prototxt", "*.prototxt.txt"])
    weights = _find_file(["*.caffemodel"])

    if prototxt is None or weights is None:
        searched = ", ".join(str(d) for d in SEARCH_DIRS)
        raise FileNotFoundError(
            "MobileNet-SSD model files not found.\n"
            f"  prototxt found:    {prototxt}\n"
            f"  caffemodel found:  {weights}\n"
            f"Searched in: {searched}\n"
            "Put the .prototxt and .caffemodel files in a 'models' folder "
            "next to app.py."
        )

    return cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))


def detect_objects(image_bgr, conf_threshold=0.8):
    """
    Run MobileNet-SSD on a BGR image.

    Returns (annotated_bgr_image, detections) where detections is a list of
    {"object": str, "confidence": float, "box": (x1, y1, x2, y2)}
    sorted by confidence, highest first.
    """
    net = _load_net()
    h, w = image_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (300, 300)),
        0.007843, (300, 300), 127.5,
    )
    net.setInput(blob)
    output = net.forward()

    annotated = image_bgr.copy()
    detections = []

    for i in range(output.shape[2]):
        confidence = float(output[0, 0, i, 2])
        if confidence < conf_threshold:
            continue

        class_id = int(output[0, 0, i, 1])
        if class_id <= 0 or class_id >= len(CLASSES):
            continue

        box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        label = CLASSES[class_id]
        detections.append({
            "object": label,
            "confidence": confidence,
            "box": (x1, y1, x2, y2),
        })

        thickness = max(2, int(round(min(h, w) / 300)))
        font_scale = max(0.5, min(h, w) / 900)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        text = f"{label}: {min(confidence * 100, 99.9):.1f}%"
        (tw, th), base = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        top = max(0, y1 - th - base - 4)
        cv2.rectangle(annotated, (x1, top), (x1 + tw + 6, top + th + base + 4),
                      (0, 255, 0), -1)
        cv2.putText(annotated, text, (x1 + 3, top + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1,
                    cv2.LINE_AA)

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return annotated, detections