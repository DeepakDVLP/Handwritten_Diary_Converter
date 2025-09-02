import streamlit as st
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

# --- Setup and Initialization ---

# Set a title for the app
st.title("Handwritten Diary OCR App")
st.write("Upload a handwritten diary page image to extract text.")

# Load PaddleOCR once and cache it for performance
@st.cache_resource
def load_ocr():
    """
    Initializes and caches the PaddleOCR model.
    This prevents the model from being reloaded every time the app reruns.
    """
    # Use_angle_cls=True helps with rotated text, lang='en' for English
    try:
        return PaddleOCR(use_angle_cls=True, lang='en')
    except Exception as e:
        st.error(f"Failed to load PaddleOCR model: {e}")
        st.stop() # Stop the app if the model cannot be loaded

# --- Main Application Logic ---

if __name__ == "__main__":
    ocr = load_ocr()

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            # Display a spinner while processing
            with st.spinner("Extracting text... This might take a moment."):
                # Load the image using PIL
                image = Image.open(uploaded_file).convert("RGB")
                
                # Display the uploaded image
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Convert the PIL image to a NumPy array for PaddleOCR
                image_np = np.array(image)

                # Run OCR on the image
                results = ocr.ocr(image_np, cls=True)

                # Check if any results were returned
                if results and results[0]:
                    # Process and concatenate the extracted text
                    extracted_text = ""
                    for line in results[0]:
                        # The OCR result is a list: [bounding_box, (text, confidence)]
                        extracted_text += line[1][0] + "\n"
                    
                    # Display the extracted text in a text area
                    st.subheader("Extracted Text")
                    st.text_area("OCR Output", extracted_text, height=300)
                else:
                    st.warning("No text was detected in the uploaded image.")

        except Exception as e:
            # Generic error handling for unexpected issues
            st.error(f"An error occurred while processing the image: {e}")
            st.warning("Please ensure the uploaded file is a valid image.")
