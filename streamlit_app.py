import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
IMG_SIZE = (150, 150)
MODEL_PATH = Path("outputs/models/best_pneumonia_model.keras")

st.set_page_config(
    page_title="Pneumonia Detection from Chest X-Ray",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load trained model: {e}")
    st.stop()

# -------------------------------------------------
# CSS — SINGLE SCREEN
# -------------------------------------------------
st.markdown("""
<style>
header[data-testid="stHeader"] {
    height: 2rem;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.4rem;
    padding-bottom: 0.4rem;
}

.main-title {
    font-size: 2.05rem;
    font-weight: 750;
    margin: 0 0 0.15rem 0;
    line-height: 1.15;
}

.subtitle {
    color: #6b7280;
    font-size: 0.95rem;
    margin: 0 0 0.75rem 0;
}

[data-testid="stFileUploader"] {
    margin-bottom: 0.25rem;
}

[data-testid="stFileUploaderDropzone"] {
    min-height: 62px;
    padding: 0.45rem 0.8rem;
}

[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}

h2, h3 {
    margin-top: 0 !important;
    margin-bottom: 0.35rem !important;
}

[data-testid="stImage"] img {
    max-height: 305px !important;
    width: 100% !important;
    object-fit: contain !important;
}

[data-testid="stImage"] {
    margin-bottom: 0 !important;
}

[data-testid="stImage"] p {
    margin-top: 0.15rem !important;
    margin-bottom: 0 !important;
}

.stButton button {
    height: 38px;
    font-weight: 600;
}

[data-testid="stMetric"] {
    padding: 0;
}

[data-testid="stMetricLabel"] {
    font-size: 0.85rem;
}

[data-testid="stMetricValue"] {
    font-size: 1.75rem;
}

[data-testid="stProgress"] {
    margin-top: -0.15rem;
    margin-bottom: 0.1rem;
}

.result-title {
    font-size: 1.25rem;
    font-weight: 700;
    padding: 0.65rem 0.85rem;
    border-radius: 8px;
    margin: 0.25rem 0 0.45rem 0;
}

.pneumonia {
    background: #fde8e8;
}

.normal {
    background: #e7f6ec;
}

.prob-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0.25rem 0 0.1rem 0;
}

.info-bar {
    margin-top: 0.5rem;
    padding: 0.55rem 0.8rem;
    border: 1px solid #d9dde3;
    border-radius: 8px;
    font-size: 0.84rem;
    color: #4b5563;
    text-align: center;
}

.disclaimer {
    margin-top: 0.35rem;
    padding: 0.45rem 0.7rem;
    border-radius: 7px;
    background: #fff8df;
    color: #806000;
    font-size: 0.80rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    '<div class="main-title">🫁 Pneumonia Detection from Chest X-Ray</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning Based Chest X-Ray Classification — '
    'upload an image to classify it as NORMAL or PNEUMONIA.</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Select a Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# MAIN AREA
# -------------------------------------------------

# Keep prediction visible across Streamlit reruns
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None


if uploaded_file is not None:

    # Unique ID for current uploaded image
    current_file = f"{uploaded_file.name}_{uploaded_file.size}"

    # New image = reset previous result
    if st.session_state.current_file != current_file:
        st.session_state.current_file = current_file
        st.session_state.analysis_result = None

    image = Image.open(uploaded_file).convert("RGB")

    left_col, right_col = st.columns([1, 1], gap="large")

    # ---------------- LEFT ----------------
    with left_col:
        st.markdown("### Chest X-Ray")

        st.image(
            image,
            caption="Uploaded Chest X-Ray",
            use_column_width=True
        )

    # ---------------- RIGHT ----------------
    with right_col:
        st.markdown("### AI Analysis")

        # Show Analyze button only BEFORE analysis
        if st.session_state.analysis_result is None:

            analyze = st.button(
                "Analyze X-Ray",
                type="primary",
                use_container_width=True
            )

            if analyze:

                with st.spinner("Analyzing..."):

                    img = image.resize(IMG_SIZE)

                    x = np.asarray(
                        img,
                        dtype=np.float32
                    ) / 255.0

                    x = np.expand_dims(
                        x,
                        axis=0
                    )

                    prediction = model.predict(
                        x,
                        verbose=0
                    )[0]

                    normal_prob = float(prediction[0])
                    pneumonia_prob = float(prediction[1])

                    if pneumonia_prob > normal_prob:
                        result = "PNEUMONIA"
                        confidence = pneumonia_prob
                    else:
                        result = "NORMAL"
                        confidence = normal_prob

                    st.session_state.analysis_result = {
                        "result": result,
                        "confidence": confidence,
                        "normal_prob": normal_prob,
                        "pneumonia_prob": pneumonia_prob
                    }

                st.rerun()

        # ---------------- RESULT ----------------
        if st.session_state.analysis_result is not None:

            data = st.session_state.analysis_result

            result = data["result"]
            confidence = data["confidence"]
            normal_prob = data["normal_prob"]
            pneumonia_prob = data["pneumonia_prob"]

            # Result shown only once
            if result == "PNEUMONIA":

                st.markdown(
                    '<div class="result-title pneumonia">'
                    'PNEUMONIA DETECTED</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="result-title normal">'
                    'NORMAL</div>',
                    unsafe_allow_html=True
                )

            # Confidence shown only once
            st.markdown(
                '<div style="font-size:0.85rem;'
                'color:#6b7280;margin-top:6px;">'
                'CONFIDENCE</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div style="font-size:2.25rem;'
                f'font-weight:750;line-height:1.05;'
                f'margin-bottom:12px;">'
                f'{confidence * 100:.2f}%</div>',
                unsafe_allow_html=True
            )

            # Distribution bars — no repeated percentages
            st.markdown(
                '<div class="prob-title">Class Distribution</div>',
                unsafe_allow_html=True
            )

            st.markdown("**NORMAL**")
            st.progress(normal_prob)

            st.markdown("**PNEUMONIA**")
            st.progress(pneumonia_prob)


# -------------------------------------------------
# COMPACT FOOTER
# -------------------------------------------------
st.markdown(
    """
    <div class="info-bar">
    <b>Model:</b> Custom CNN &nbsp; • &nbsp;
    <b>Input:</b> 150 × 150 &nbsp; • &nbsp;
    <b>Classes:</b> NORMAL / PNEUMONIA &nbsp; • &nbsp;
    <b>Test Accuracy:</b> 91.51%
    <br>
    <span style="font-size:0.78rem;">
    <b>Confusion Matrix:</b>
    TN = 196 &nbsp; | &nbsp;
    FP = 38 &nbsp; | &nbsp;
    FN = 15 &nbsp; | &nbsp;
    TP = 375
    </span>
    </div>

    <div class="disclaimer">
    For educational and research purposes only — not intended for clinical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)
