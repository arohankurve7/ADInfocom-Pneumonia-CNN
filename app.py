import tkinter as tk
from pathlib import Path
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

MODEL_PATH = "outputs/models/best_pneumonia_model.keras"
IMG_SIZE = (150, 150)

print("Loading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

root = tk.Tk()
root.title("Pneumonia Detection from Chest X-Ray")
root.geometry("900x780")

title = tk.Label(
    root,
    text="Pneumonia Detection from Chest X-Ray",
    font=("Arial", 24, "bold")
)
title.pack(pady=(25, 5))

subtitle = tk.Label(
    root,
    text="Deep Learning Based Chest X-Ray Classification",
    font=("Arial", 14)
)
subtitle.pack(pady=(0, 15))

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(
    root,
    text="Select a test X-Ray",
    font=("Arial", 22, "bold")
)
result_label.pack(pady=8)

confidence_label = tk.Label(root, text="", font=("Arial", 16))
confidence_label.pack()

probability_label = tk.Label(root, text="", font=("Arial", 14))
probability_label.pack(pady=8)


def predict_path(path):
    path = str(path)

    display_img = Image.open(path).convert("RGB")
    display_img.thumbnail((500, 340))
    photo = ImageTk.PhotoImage(display_img)

    image_label.configure(image=photo)
    image_label.image = photo

    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)

    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x, verbose=0)[0]

    normal_prob = float(prediction[0])
    pneumonia_prob = float(prediction[1])

    if pneumonia_prob > normal_prob:
        result = "PNEUMONIA"
        confidence = pneumonia_prob
    else:
        result = "NORMAL"
        confidence = normal_prob

    result_label.config(text=f"Prediction: {result}")
    confidence_label.config(
        text=f"Confidence: {confidence * 100:.2f}%"
    )
    probability_label.config(
        text=(
            f"NORMAL: {normal_prob * 100:.2f}%    |    "
            f"PNEUMONIA: {pneumonia_prob * 100:.2f}%"
        )
    )


def test_pneumonia():
    files = sorted(Path("chest_xray/test/PNEUMONIA").glob("*"))
    if files:
        predict_path(files[0])


def test_normal():
    files = sorted(Path("chest_xray/test/NORMAL").glob("*"))
    if files:
        predict_path(files[0])


button_frame = tk.Frame(root)
button_frame.pack(pady=15)

p_button = tk.Button(
    button_frame,
    text="Test PNEUMONIA X-Ray",
    command=test_pneumonia,
    font=("Arial", 14, "bold"),
    padx=18,
    pady=10
)
p_button.pack(side="left", padx=10)

n_button = tk.Button(
    button_frame,
    text="Test NORMAL X-Ray",
    command=test_normal,
    font=("Arial", 14, "bold"),
    padx=18,
    pady=10
)
n_button.pack(side="left", padx=10)

disclaimer = tk.Label(
    root,
    text="For educational and research use only — not a clinical diagnosis.",
    font=("Arial", 11)
)
disclaimer.pack(side="bottom", pady=15)

root.mainloop()
