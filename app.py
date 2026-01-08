import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

# Page config
st.set_page_config(page_title="YOLOv11 Image Detection", layout="centered")

st.title("🧠 YOLOv11 Object Detection")
st.write("Upload an image to detect objects using YOLOv11n")

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect Objects"):
        with st.spinner("Detecting objects..."):
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Save temp image
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, img_bgr)
                temp_path = tmp.name

            # Run YOLO
            results = model(temp_path)

            # Get annotated image
            annotated_img = results[0].plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            # Show result
            st.image(
                annotated_img,
                caption="Detection Result",
                use_column_width=True
            )

            # Show detection details
            st.subheader("📊 Detection Details")
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                st.write(f"**{label}** — Confidence: `{conf:.2f}`")


