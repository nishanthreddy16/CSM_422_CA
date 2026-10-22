"""
PlantGuard AI - web app.

Run:  streamlit run app.py
"""
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import config
import utils
from disease_info import GENERAL_TIPS, get_info
from gradcam import make_gradcam, overlay_heatmap

st.set_page_config(page_title="PlantGuard AI", page_icon="🌿", layout="wide")

st.markdown("""
<style>
.big-title {font-size: 2.4rem; font-weight: 800; color: #2e7d32; margin-bottom: 0;}
.subtitle {color: #666; margin-bottom: 1.2rem;}
.card {padding: 1rem 1.2rem; border-radius: 12px; margin-bottom: 0.8rem;}
.healthy {background: #e8f5e9; border-left: 6px solid #2e7d32; color: #1b5e20;}
.diseased {background: #fff3e0; border-left: 6px solid #ef6c00; color: #5d3000;}
.unknown {background: #eceff1; border-left: 6px solid #607d8b; color: #263238;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🌿 PlantGuard AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Plant disease detection using a Convolutional Neural Network</p>',
            unsafe_allow_html=True)


@st.cache_resource
def load_everything():
    return utils.load_model(), utils.load_class_names()


try:
    model, class_names = load_everything()
except Exception:
    st.error("Trained model not found. Please run `python train.py` first, then restart the app.")
    st.stop()

img_size = model.input_shape[1]

with st.sidebar:
    st.header("About")
    st.write("PlantGuard AI looks at a photo of a leaf and predicts whether the plant is "
             "healthy or which disease it has, then suggests treatment and prevention.")
    st.write(f"**Classes the model knows ({len(class_names)}):**")
    for c in class_names:
        st.write("•", utils.pretty_name(c))
    st.caption(GENERAL_TIPS)

tab_diag, tab_perf = st.tabs(["🔍 Diagnose", "📊 Model performance"])

# ---------------------------------------------------------------- Diagnose
with tab_diag:
    source = st.radio("Image source", ["Upload a file", "Use camera"], horizontal=True)
    file = (st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
            if source == "Upload a file" else st.camera_input("Take a photo of the leaf"))

    if file:
        pil_img = Image.open(file).convert("RGB")
        batch = utils.prepare_image(pil_img, img_size)
        probs = model.predict(batch, verbose=0)[0]
        top = np.argsort(probs)[::-1][:3]
        best = class_names[top[0]]
        conf = float(probs[top[0]])
        info = get_info(best)

        left, right = st.columns(2)
        with left:
            st.image(pil_img, caption="Uploaded leaf", width="stretch")
        with right:
            css = {"Healthy": "healthy", "Diseased": "diseased"}.get(info["status"], "unknown")
            icon = "✅" if info["status"] == "Healthy" else "⚠️"
            st.markdown(
                f'<div class="card {css}"><h3 style="margin:0">{icon} '
                f'{utils.pretty_name(best)}</h3>'
                f'Status: <b>{info["status"]}</b> &nbsp;|&nbsp; Confidence: <b>{conf*100:.1f}%</b></div>',
                unsafe_allow_html=True)
            if conf < config.CONFIDENCE_THRESHOLD:
                st.warning("The model is not very sure. Try a clearer, close-up photo of a single leaf.")
            st.markdown(f"**About:** {info['description']}")
            st.markdown(f"**General guidance (not a confirmed diagnosis):** {info['treatment']}")
            st.markdown(f"**Prevention tips:** {info['prevention']}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top 3 predictions")
            df = pd.DataFrame({"Class": [utils.pretty_name(class_names[i]) for i in top],
                               "Probability (%)": [float(probs[i]) * 100 for i in top]}).set_index("Class")
            st.bar_chart(df)
        with c2:
            st.subheader("Where did the AI look? (Grad-CAM)")
            try:
                cam, _ = make_gradcam(model, batch)
                st.image(overlay_heatmap(pil_img, cam), width="stretch",
                         caption="Red/yellow = areas that influenced the decision most")
            except Exception as e:
                st.info(f"Grad-CAM not available: {e}")
        st.caption(GENERAL_TIPS)

# ---------------------------------------------------------------- Performance
with tab_perf:
    metrics_file = config.OUTPUT_DIR / "metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        a, b, c = st.columns(3)
        a.metric("Validation accuracy", f"{m['validation_accuracy'] * 100:.2f}%")
        b.metric("Number of classes", len(m["classes"]))
        c.metric("Epochs run", m["epochs_run"])
    for fname, title in [("training_curves.png", "Training curves"),
                         ("confusion_matrix.png", "Confusion matrix"),
                         ("sample_predictions.png", "Sample predictions"),
                         ("class_distribution.png", "Class distribution")]:
        p = config.OUTPUT_DIR / fname
        if p.exists():
            st.subheader(title)
            st.image(str(p), width="stretch")
    report = config.OUTPUT_DIR / "classification_report.txt"
    if report.exists():
        st.subheader("Classification report")
        st.code(report.read_text())
