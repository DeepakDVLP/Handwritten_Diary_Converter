import streamlit as st
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

# Load PaddleOCR once
@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

ocr = load_ocr()

st.title("Handwritten Diary OCR App")
st.write("Upload a handwritten diary page image to extract text")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Load image using PIL
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to NumPy array
        image_np = np.array(image)

        # Run OCR
        with st.spinner("Extracting text..."):
            results = ocr.ocr(image_np, cls=True)

        # Display results
        st.subheader("Extracted Text")
        extracted_text = ""
        for line in results[0]:
            extracted_text += line[1][0] + "\n"

        st.text_area("OCR Output", extracted_text, height=200)

    except Exception as e:
        st.error(f"Error processing image: {e}")
