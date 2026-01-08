import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Page settings
st.set_page_config(
    page_title="YOLOv11 Image Detection",
    layout="wide"
)

st.title("🧠 YOLOv11 Object Detection")
st.write("Upload an image and detect objects using YOLOv11")

# Load model once
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Upload image
uploaded_image = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_image:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    # Show original image
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    # Run YOLO detection
    results = model(image_path)

    # Plot detection image
    detected_image = results[0].plot()

    # Show detected image
    with col2:
        st.subheader("✅ Detected Image")
        st.image(detected_image, use_column_width=True)

    # Show detected objects
    st.subheader("📊 Detection Details")
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        st.write(f"**{class_name}** → Confidence: `{confidence:.2f}`")

    os.remove(image_path)
