import streamlit as st
import numpy as np
import cv2
from PIL import Image

from segmentation_utils import get_crop_coords, get_mask, apply_mask_transparent
from resnet_50_inference import infer_from_resnet50

# -------------------------------
# Streamlit UI Config
# -------------------------------
st.set_page_config(
    page_title="Mammal-Vision-v1",
    layout="wide"
)

st.title("🐒🐘🐂 Mammal-Vision-v1: GrabCut + ResNet50 Mammal Recognition")

def resnet_50():
    img_path = 'model_input/input_crop.png'
    img = Image.open(img_path).convert('RGB')
    idx = infer_from_resnet50(img)

    # mapping it to a class label
    with open('labels.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    label_map = {i: name for i, name in enumerate(classes)}
    return label_map[idx]

def full_pipeline(image):
    coords = get_crop_coords(image)
    binary_mask = get_mask(image, coords)
    model_input = apply_mask_transparent(image, binary_mask, coords)
    cv2.imwrite("model_input/input_crop.png", model_input)
    output_label = resnet_50()
    return output_label


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("Uploaded Mammal Scene:")

    # Compact display – width controlled
    st.image(img, width=630)

    if st.button("Know your mammal"):
        if full_pipeline(img) is not None:
            label = full_pipeline(img)

            op = f"""
            <div style="
                border:1px solid #888;
                padding:12px;
                border-radius:8px;
                font-size:20px;
                font-weight:600;
                width:fit-content;">
            {label}
            </div>
            """

            st.markdown(op, unsafe_allow_html=True)

            