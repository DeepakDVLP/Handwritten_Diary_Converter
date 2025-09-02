import streamlit as st
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

st.set_page_config(page_title="Cheating Detection App", layout="wide")

st.title("Cheating Detection System")
st.write("Upload an image or video to detect cheating behavior.")

# Upload options
option = st.selectbox("Choose Input Type", ["Image", "Video"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Run OCR for text detection
        results = ocr.ocr(np.array(image), cls=True)
        detected_text = []
        for res in results:
            for line in res:
                text, conf = line[1][0], line[1][1]
                detected_text.append(f"{text} (conf: {conf:.2f})")
        
        st.subheader("Detected Text:")
        if detected_text:
            st.write("\n".join(detected_text))
        else:
            st.write("No text detected.")

elif option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        st.video(uploaded_video)
        st.info("Video processing feature can be expanded for real-time cheating detection.")
