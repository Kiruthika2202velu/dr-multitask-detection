"""
app.py
Streamlit demo for the Explainable Multi-Task DR model.

Upload a fundus (retina) image -> get:
  - Predicted DR stage (No DR / Mild / Moderate / Severe / Proliferative)
  - Blindness risk probability (High / Low)
  - Grad-CAM heatmap showing which regions drove the prediction

Run locally:
    streamlit run app.py

Requires a trained model at checkpoints/best_model.keras
(produced by src/train.py). No GPU needed for inference.
"""

import sys
import os
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from gradcam import make_gradcam_heatmap, overlay_heatmap
from utils import format_prediction

IMG_SIZE = 300
MODEL_PATH = "checkpoints/best_model.keras"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"No trained model found at {MODEL_PATH}. Run src/train.py first.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    return img.astype(np.float32) / 255.0


def main():
    st.set_page_config(page_title="DR Multi-Task Detector", layout="wide")
    st.title("Explainable Diabetic Retinopathy Stage & Blindness Risk Detector")
    st.caption(
        "Multi-task deep learning: EfficientNetB3 backbone + DR stage classification "
        "+ blindness risk scoring, with Grad-CAM explainability."
    )

    model = load_model()

    uploaded_file = st.file_uploader("Upload a retina fundus image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        processed = preprocess_image(pil_img)
        batch = np.expand_dims(processed, axis=0)

        stage_pred, risk_pred = model.predict(batch, verbose=0)
        result = format_prediction(stage_pred[0], risk_pred[0][0])

        heatmap = make_gradcam_heatmap(model, batch, output_name="stage_output",
                                        class_index=result["stage_index"])
        overlay = overlay_heatmap(processed, heatmap)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(processed, use_container_width=True)
        with col2:
            st.subheader("Grad-CAM (why the model decided this)")
            st.image(overlay, use_container_width=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted DR Stage", result["stage_name"],
                      f"{result['stage_confidence']*100:.1f}% confidence")
            st.bar_chart(result["stage_probs"])
        with c2:
            st.metric("Blindness Risk", result["risk_name"],
                      f"{result['risk_probability']*100:.1f}% probability")
            if result["risk_probability"] >= 0.5:
                st.warning("This image is flagged as HIGH risk — recommend urgent ophthalmologist review.")
            else:
                st.success("This image is flagged as LOW risk — routine follow-up recommended.")

        st.caption(
            "This tool is a research/thesis prototype, not a certified medical device. "
            "All predictions should be verified by a qualified ophthalmologist."
        )
    else:
        st.info("Upload a fundus image to get a prediction (try a sample from the APTOS/IDRiD validation set).")


if __name__ == "__main__":
    main()
