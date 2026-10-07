import cv2

from modules.object_detection import detect_objects


# Load test image
image = cv2.imread("input/clean_test.png")

if image is None:
    print("❌ Could not load test image")
    exit()

print("✅ Image loaded")


# Detect objects
annotated_image, detections = detect_objects(image)


print("\n========== OBJECT DETECTION ==========")


if detections:

    for detection in detections:

        print(
            f"Detected: {detection['object']} "
            f"| Confidence: "
            f"{detection['confidence'] * 100:.1f}%"
        )

else:

    print(
        "No objects detected with confidence >= 80%."
    )


print("======================================")


# Save result
cv2.imwrite(
    "output/detection/detection_result.png",
    annotated_image
)


print("\n✅ Result saved to:")
print("output/detection/detection_result.png")