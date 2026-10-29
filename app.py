import hashlib
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from ultralytics import YOLO

st.set_page_config(page_title="CocoNexa", page_icon="🥥", layout="wide")

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")

CLASS_INFO = {
    "Young": {
        "emoji": "🟢",
        "color": "#2e7d32",
        "desc": "Green husk, plenty of sweet water, very soft thin meat.",
        "use": "Best for drinking water.",
    },
    "Semi-mature": {
        "emoji": "🟡",
        "color": "#9e9d24",
        "desc": "Husk turning yellow-green, water reducing, meat thickening.",
        "use": "Good for tender coconut meat and milk.",
    },
    "Mature": {
        "emoji": "🟤",
        "color": "#8d6e63",
        "desc": "Brown husk, less water, firm white meat.",
        "use": "Best for oil, grating and cooking.",
    },
    "Over mature": {
        "emoji": "⚫",
        "color": "#4e342e",
        "desc": "Dry dark husk, little water, hard meat.",
        "use": "Mostly used for seed or copra, not for fresh eating.",
    },
}
CLASS_NAMES = list(CLASS_INFO.keys())


def normalize(name):
    """Map whatever label is inside best.pt to one of the four display classes."""
    compact = str(name).lower().replace("_", "").replace("-", "").replace(" ", "")
    for cls in CLASS_NAMES:
        if compact == cls.lower().replace("-", "").replace(" ", ""):
            return cls
    return str(name)


@st.cache_resource(show_spinner="Loading CocoNexa model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None  # demo mode
    return YOLO(MODEL_PATH)


def run_detection(model, image, conf, sig):
    if model is None:
        # Demo mode: placeholder label chosen from the photo, clearly marked
        label = CLASS_NAMES[int(sig, 16) % len(CLASS_NAMES)]
        demo = image.copy()
        ImageDraw.Draw(demo).text((10, 10), f"DEMO: {label} (no best.pt)", fill=(255, 0, 0))
        return [(label, 1.0)], demo
    # Ultralytics expects BGR numpy arrays, so flip RGB -> BGR
    bgr = np.array(image)[:, :, ::-1]
    result = model.predict(bgr, conf=conf, verbose=False)[0]
    detections = []
    for box in result.boxes:
        label = normalize(result.names[int(box.cls[0])])
        detections.append((label, float(box.conf[0])))
    annotated = Image.fromarray(result.plot()[:, :, ::-1])
    return detections, annotated


def speak(text):
    """Voice assistant output: the browser reads the text aloud."""
    components.html(
        f"""
        <script>
          const text = {json.dumps(text)};
          window.speechSynthesis.cancel();
          const u = new SpeechSynthesisUtterance(text);
          u.lang = "en-US";
          u.rate = 0.95;
          window.speechSynthesis.speak(u);
        </script>
        """,
        height=0,
    )


if "counts" not in st.session_state:
    st.session_state.counts = {c: 0 for c in CLASS_NAMES}
if "last_sig" not in st.session_state:
    st.session_state.last_sig = None
if "latest" not in st.session_state:
    st.session_state.latest = None

# Sidebar is created first so it can be filled after the detection runs
sidebar = st.sidebar.container()
with sidebar:
    st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence threshold", 0.10, 0.90, 0.40, 0.05)

st.title("🥥 CocoNexa")
st.caption("Point the camera at a coconut or upload a photo. CocoNexa tells you its maturity stage.")

model = load_model()
if model is None:
    st.warning(
        f"Demo mode: '{MODEL_PATH}' not found. Results are placeholders, not real maturity detection. "
        "Add your best.pt to the project folder to turn on real detection."
    )

tab_cam, tab_up = st.tabs(["📷 Camera", "🖼️ Upload"])
with tab_cam:
    camera_photo = st.camera_input("Take a picture of a coconut")
with tab_up:
    uploaded = st.file_uploader("Upload a coconut photo", type=["jpg", "jpeg", "png"])

source = uploaded or camera_photo

if source is None:
    st.info("Take a photo or upload one to start.")
else:
    raw = source.getvalue()
    sig = hashlib.md5(raw).hexdigest()
    image = Image.open(source).convert("RGB")

    with st.spinner("Detecting maturity..."):
        detections, annotated = run_detection(model, image, conf, sig)
    prefix = "Demo mode. " if model is None else ""

    # Count each new photo once, not again on every rerun
    if sig != st.session_state.last_sig:
        st.session_state.last_sig = sig
        if detections:
            best_label = max(detections, key=lambda d: d[1])[0]
            st.session_state.latest = best_label
            for label, _ in detections:
                if label in st.session_state.counts:
                    st.session_state.counts[label] += 1
            info = CLASS_INFO.get(best_label)
            if info:
                speak(f"{prefix}This coconut is {best_label}. {info['desc']} {info['use']}")
        else:
            st.session_state.latest = None
            speak(f"{prefix}No coconut detected. Please try a clearer photo.")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(annotated, caption="Detection result", use_container_width=True)
    with col2:
        latest = st.session_state.latest
        if latest is None:
            st.warning("No coconut detected. Try a clearer photo or lower the threshold.")
        else:
            info = CLASS_INFO[latest]
            st.success(f"{info['emoji']} **{latest}**")
            st.write(info["desc"])
            st.write(f"**Use:** {info['use']}")
        if st.button("🔊 Read result aloud"):
            if latest:
                speak(f"{latest}. {CLASS_INFO[latest]['desc']} {CLASS_INFO[latest]['use']}")

# Voice assistant: explain all classes on demand
st.divider()
if st.button("🎙️ Explain the four maturity stages"):
    speak(" ".join(f"{n}. {CLASS_INFO[n]['desc']}" for n in CLASS_NAMES))

# Dashboard in the sidebar
with sidebar:
    st.header("🥥 Maturity Dashboard")
    counts = st.session_state.counts
    latest = st.session_state.latest
    for name in CLASS_NAMES:
        info = CLASS_INFO[name]
        border = f"3px solid {info['color']}" if name == latest else "1px solid #ddd"
        st.markdown(
            f"""
            <div style="border:{border};border-radius:10px;padding:8px 10px;margin:6px 0">
              <b>{info['emoji']} {name}</b> &nbsp; <span style="float:right">{counts[name]}</span><br>
              <small>{info['desc']}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.subheader("Coconuts counted")
    st.bar_chart(pd.DataFrame({"count": counts}))
    if st.button("Reset counts"):
        st.session_state.counts = {c: 0 for c in CLASS_NAMES}
        st.session_state.latest = None
        st.session_state.last_sig = None
        st.rerun()
