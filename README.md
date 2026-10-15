# Pneumonia Detection from Chest X-Ray Images using Deep Learning

A deep learning-based research and educational project for binary classification of chest X-ray images into **NORMAL** and **PNEUMONIA** categories.

This implementation includes a custom Convolutional Neural Network (CNN), model training and evaluation scripts, a prediction utility, a Streamlit dashboard, and a Tkinter desktop interface.

## Project Overview

| Component | Description |
|---|---|
| Domain | Deep Learning, Computer Vision, Medical Image Analysis |
| Task | Binary chest X-ray classification |
| Classes | NORMAL, PNEUMONIA |
| Model | Custom CNN |
| Framework | TensorFlow / Keras |
| Input Resolution | 150 × 150 pixels |
| User Interface | Streamlit and Tkinter |
| Python Version | 3.10 |

## Dataset

**Chest X-Ray Images (Pneumonia)**

- Training: 5,216 images
- Validation: 16 images
- Testing: 624 images
- Total: 5,856 images

Dataset source: [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Original research: Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell (2018).

The dataset is not included in this repository.

Expected directory structure:

    chest_xray/
      train/
        NORMAL/
        PNEUMONIA/
      val/
        NORMAL/
        PNEUMONIA/
      test/
        NORMAL/
        PNEUMONIA/

## Model Architecture

The implementation uses a custom sequential CNN with:

- 10 convolutional layers
- 5 max-pooling layers
- Fully connected classification layers
- Dropout regularization (0.4)
- Binary classification output

Training configuration:

| Parameter | Value |
|---|---|
| Image size | 150 × 150 |
| Batch size | 163 |
| Maximum epochs | 100 |
| Completed epochs | 20 |
| Random seed | 42 |

Training incorporates class weighting, model checkpointing, early stopping, and learning-rate reduction.

## Experimental Results

Evaluated on **624 held-out test images**.

| Metric | Value |
|---|---|
| Test accuracy | **91.51%** |
| Test loss | 0.2741 |
| Pneumonia precision | 90.80% |
| Pneumonia recall | **96.15%** |
| Pneumonia F1-score | 93.40% |
| Normal precision | 92.89% |
| Normal recall | 83.76% |
| Normal F1-score | 88.09% |
| Weighted F1-score | 91.41% |

These results describe this implementation's saved evaluation, not the results reported by the original upstream project.

The evaluation is limited to the specified test dataset and does not establish clinical diagnostic performance.

## Project Files

| File | Purpose |
|---|---|
| `train_model.py` | Custom CNN training and evaluation |
| `predict.py` | Single-image prediction |
| `streamlit_app.py` | Interactive web dashboard |
| `app.py` | Tkinter desktop interface |
| `requirements.txt` | Python dependencies |
| `code/` | Original upstream notebooks and reference implementations |

## Installation

Create a Python 3.10 virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

## Training

Place the dataset inside `chest_xray/`, then run:

```bash
.venv/bin/python train_model.py
```

The training script saves models, evaluation reports and plots under `outputs/`.

## Streamlit Dashboard

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

The dashboard requires the trained model at:

`outputs/models/best_pneumonia_model.keras`

## Prediction and Desktop Application

Single-image prediction:

```bash
.venv/bin/python predict.py path/to/image.jpeg
```

Tkinter desktop application:

```bash
.venv/bin/python app.py
```

## Original Project and Attribution

This project builds upon the open-source work of **Anjana Tiha**.

Original repository:
https://github.com/anjanatiha/Pneumonia-Detection-from-Chest-X-Ray-Images-with-Deep-Learning

The original notebooks are retained under `code/`. The custom training, prediction and application scripts are documented separately.

Original copyright: Copyright (c) 2018 Anjana Tiha.

The original MIT License is preserved in `LICENSE`.

## Disclaimer

This project is intended for educational and research purposes only. It is not clinically validated and must not be used independently for medical diagnosis.
