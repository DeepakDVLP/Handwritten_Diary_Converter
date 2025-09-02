import streamlit as st
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR, PPStructure, save_structure_res

# Set Streamlit page configuration
st.set_page_config(page_title="Handwritten Diary OCR App", layout="wide")

st.title("Handwritten Diary OCR App")
st.write("Upload a handwritten diary page image to extract text from the table.")

# Load PaddleOCR's Structure model once and cache it
@st.cache_resource
def load_ppstructure_model():
    """
    Initializes and caches the PPStructure model for table recognition.
    """
    st.info("Loading PPStructure model... This may take a moment.")
    try:
        # Use PPStructure for advanced document analysis including table recognition
        return PPStructure(show_log=True, lang='en', use_angle_cls=True)
    except Exception as e:
        st.error(f"Failed to load PPStructure model: {e}")
        return None

structure_engine = load_ppstructure_model()

if structure_engine is None:
    st.warning("Application cannot run without the PPStructure model.")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Load image using PIL
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Run table recognition
        with st.spinner("Extracting table data..."):
            # The structure engine processes the image and returns a list of dictionaries
            # containing text, tables, and other elements.
            result = structure_engine(np.array(image))

        st.subheader("Extracted Table Data")

        # Process the results
        extracted_text = ""
        found_table = False
        
        for line in result:
            # Check for table objects in the result
            if line['type'] == 'table':
                found_table = True
                # The 'html' key contains the table data in HTML format
                extracted_text += "--- Table Found ---\n"
                extracted_text += line['res']['html']
                extracted_text += "\n"
            else:
                # Optionally, extract regular text outside of tables
                extracted_text += line['res']['text'] + "\n"
        
        if found_table:
            st.success("Successfully extracted table structure and data!")
            st.text_area("Table HTML Output", extracted_text, height=400)
            
            # --- Bonus: Show a preview of the HTML table ---
            # You can also display the HTML directly in Streamlit
            st.subheader("Table Preview")
            st.markdown(extracted_text, unsafe_allow_html=True)
        else:
            st.warning("No tables were detected in the uploaded image.")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
        st.warning("Please ensure the uploaded file is a valid image containing a readable table.")
