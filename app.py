import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

st.set_page_config(page_title="YOLOv11 Detection", layout="centered")

st.title("🧠 YOLOv11 Image Object Detection")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Objects"):
        with st.spinner("Running YOLO detection..."):
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, img_bgr)
                results = model(tmp.name)

            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            st.image(annotated, caption="Detection Result", use_column_width=True)

            st.subheader("📊 Detected Objects")
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                st.write(f"**{model.names[cls]}** — {conf:.2f}")
