
import streamlit as st
import easyocr
import pandas as pd
from PIL import Image

st.title("Handwritten Diary to Digital Table Converter")

# File uploader for image
uploaded_file = st.file_uploader("Upload a diary image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Initialize OCR Reader
    reader = easyocr.Reader(['en'])
    result = reader.readtext(uploaded_file)

    # Extract text
    extracted_text = [res[1] for res in result]
    df = pd.DataFrame(extracted_text, columns=["Extracted Text"])

    st.subheader("Extracted Text")
    st.dataframe(df)

    # Downloadable CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="extracted_text.csv",
        mime="text/csv"
    )
