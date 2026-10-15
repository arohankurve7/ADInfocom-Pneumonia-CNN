from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("outputs/models/best_pneumonia_model.keras")
IMG_SIZE = (150, 150)

if len(sys.argv) != 2:
    print("\nUsage:")
    print("python predict.py <xray_image_path>")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.exists():
    print(f"\nERROR: Image not found: {image_path}")
    sys.exit(1)

if not MODEL_PATH.exists():
    print(f"\nERROR: Model not found: {MODEL_PATH}")
    sys.exit(1)

print("\n==========================================")
print("       PNEUMONIA X-RAY PREDICTION")
print("==========================================")

print(f"\nLoading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Analyzing image: {image_path.name}")

img = Image.open(image_path).convert("RGB")
img = img.resize(IMG_SIZE)

x = np.asarray(img, dtype=np.float32) / 255.0
x = np.expand_dims(x, axis=0)

prediction = model.predict(x, verbose=0)[0]

normal_prob = float(prediction[0])
pneumonia_prob = float(prediction[1])

if pneumonia_prob > normal_prob:
    predicted_class = "PNEUMONIA"
    confidence = pneumonia_prob
else:
    predicted_class = "NORMAL"
    confidence = normal_prob

print("\n------------------------------------------")
print("              PREDICTION")
print("------------------------------------------")
print(f"Result               : {predicted_class}")
print(f"Confidence           : {confidence * 100:.2f}%")
print(f"NORMAL probability   : {normal_prob * 100:.2f}%")
print(f"PNEUMONIA probability: {pneumonia_prob * 100:.2f}%")
print("------------------------------------------")
print("\nFor educational/research use only.")
print("This output is not a clinical diagnosis.")
print("==========================================\n")
