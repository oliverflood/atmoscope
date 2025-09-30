import os, sys
# add current directory to top of Python interpreter's module search path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
from PIL import Image, ImageOps
from clouds.serving.model_service import CloudClassifier

st.set_page_config(page_title="Cloud Classifier", page_icon="☁️", layout="centered") # emoji in code spotted!
st.title("Cloud Classifier")

MODEL_PATH = os.getenv("CLOUDS_MODEL_PATH", "models/clouds_bundle.pt")

@st.cache_resource
def load_service():
    return CloudClassifier(MODEL_PATH, device=None)

svc = load_service()

uploaded = st.file_uploader("Upload a cloud photo", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    img = ImageOps.exif_transpose(img).convert("RGB") # type: ignore

    st.image(img)

    probs, preds = svc.predict(img)

    TOPK = min(7, len(svc.classes))
    rows = sorted(zip(svc.classes, probs, preds), key=lambda x: x[1], reverse=True)[:TOPK]

    st.subheader("Predictions")
    st.dataframe(
        {
            "class": [r[0] for r in rows],
            "prob":  [round(float(r[1]), 3) for r in rows],
            "pred":  [bool(r[2]) for r in rows],
        },
        hide_index=True
    )
else:
    st.info("Upload an image to get predictions.")