from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("chest_xray")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (150, 150)

# Original final notebook used 163 here.
BATCH_SIZE = 163

# Final notebook training cell used 100 epochs.
# EarlyStopping normally stops before this if validation stops improving.
EPOCHS = 100

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# DATA GENERATORS
# ============================================================

print("\n==========================================")
print("       PNEUMONIA DETECTION PROJECT")
print("==========================================")

print("\nLoading dataset...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.15,
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.15,
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
    subset="training",
)

validation_generator = validation_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=64,
    class_mode="categorical",
    shuffle=False,
    seed=SEED,
    subset="validation",
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(train_generator.classes)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_generator.classes,
)

class_weights = dict(zip(classes, weights))

print("\nClass indices:", train_generator.class_indices)
print("Class weights:", class_weights)


# ============================================================
# ORIGINAL CUSTOM CNN - TF 2.x COMPATIBLE
# ============================================================

def get_conv_model():

    model = Sequential(name="Pneumonia_Custom_CNN")

    model.add(
        Conv2D(
            16,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(150, 150, 3),
        )
    )

    model.add(
        Conv2D(
            16,
            (3, 3),
            padding="same",
            activation="relu",
        )
    )

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
        )
    )

    model.add(
        Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu",
        )
    )

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same",
        )
    )

    model.add(
        Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu",
        )
    )

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2D(
            96,
            (3, 3),
            dilation_rate=(2, 2),
            activation="relu",
            padding="same",
        )
    )

    model.add(
        Conv2D(
            96,
            (3, 3),
            padding="valid",
            activation="relu",
        )
    )

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2D(
            128,
            (3, 3),
            dilation_rate=(2, 2),
            activation="relu",
            padding="same",
        )
    )

    model.add(
        Conv2D(
            128,
            (3, 3),
            padding="valid",
            activation="relu",
        )
    )

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(
        Dense(
            64,
            activation="relu",
        )
    )

    model.add(
        Dropout(0.4)
    )

    model.add(
        Dense(
            2,
            activation="softmax",
        )
    )

    return model


model = get_conv_model()

print("\nMODEL ARCHITECTURE")
print("==========================================")

model.summary()


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    str(MODEL_DIR / "best_pneumonia_model.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    verbose=1,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    patience=4,
    cooldown=2,
    min_lr=1e-10,
    verbose=1,
)

callbacks = [
    checkpoint,
    reduce_lr,
    early_stopping,
]


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n==========================================")
print("             TRAINING STARTED")
print("==========================================\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = MODEL_DIR / "final_pneumonia_model.keras"

model.save(final_model_path)

print("\nFinal model saved:")
print(final_model_path)


# ============================================================
# TRAINING CURVES
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()

accuracy_plot = PLOT_DIR / "accuracy_curve.png"

plt.savefig(
    accuracy_plot,
    dpi=300,
)

plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()

loss_plot = PLOT_DIR / "loss_curve.png"

plt.savefig(
    loss_plot,
    dpi=300,
)

plt.close()


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n==========================================")
print("             TEST EVALUATION")
print("==========================================\n")

test_generator.reset()

test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1,
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# PREDICTIONS
# ============================================================

test_generator.reset()

predictions = model.predict(
    test_generator,
    verbose=1,
)

predicted_classes = np.argmax(
    predictions,
    axis=1,
)

true_classes = test_generator.classes

class_names = list(
    test_generator.class_indices.keys()
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names,
    digits=4,
)

print("\nCLASSIFICATION REPORT")
print("------------------------------------------")
print(report)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
) as f:

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes,
)

print("\nCONFUSION MATRIX")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

disp.plot(
    cmap="Blues",
    values_format="d",
)

plt.title(
    "Pneumonia Detection - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "confusion_matrix.png",
    dpi=300,
)

plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "class_indices": train_generator.class_indices,
    "class_weights": {
        str(k): float(v)
        for k, v in class_weights.items()
    },
    "epochs_completed": len(history.history["loss"]),
}

with open(
    OUTPUT_DIR / "results.json",
    "w",
) as f:

    json.dump(
        results,
        f,
        indent=4,
    )


print("\n==========================================")
print("             PROJECT COMPLETE")
print("==========================================")

print(f"""
Test Accuracy : {test_accuracy * 100:.2f}%
Test Loss     : {test_loss:.4f}

Generated files:

{MODEL_DIR}/best_pneumonia_model.keras
{MODEL_DIR}/final_pneumonia_model.keras

{PLOT_DIR}/accuracy_curve.png
{PLOT_DIR}/loss_curve.png
{PLOT_DIR}/confusion_matrix.png

{OUTPUT_DIR}/classification_report.txt
{OUTPUT_DIR}/results.json
""")