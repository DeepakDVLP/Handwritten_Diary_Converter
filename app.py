import streamlit as st
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

# Load PaddleOCR once
@st.cache_resource
def load_ocr():
    # The use_angle_cls=True parameter is passed during model initialization
    # where it is correctly recognized.
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
            # The 'cls' parameter is not needed here.
            # Angle classification is handled by the model initialized with `use_angle_cls=True`.
            results = ocr.ocr(image_np)

        # Display results
        st.subheader("Extracted Text")
        extracted_text = ""
        # The result format is a list of lists. We need to handle the possibility of no results.
        if results and results[0]:
            for line in results[0]:
                extracted_text += line[1][0] + "\n"
        else:
            extracted_text = "No text found."

        st.text_area("OCR Output", extracted_text, height=200)

    except Exception as e:
        st.error(f"Error processing image: {e}")
