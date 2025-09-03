import streamlit as st
import pandas as pd
from paddleocr import PaddleOCR
import os

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

st.set_page_config(page_title="Handwritten Diary OCR Tool", layout="centered")
st.title("📄 Handwritten Diary OCR Tool")

# File uploader
uploaded_file = st.file_uploader("Upload an image (Driver's Diary or Arms Register)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save uploaded file
    img_path = "uploaded.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show preview image
    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    # Run OCR on button click
    if st.button("Extract Table"):
        with st.spinner("Processing..."):
            result = ocr.ocr(img_path, cls=True)
            text_data = [line[1][0] for line in result[0]]

            # Decide row size dynamically for both formats
            row_size = 6 if len(text_data) > 30 else 5
            rows = [text_data[i:i + row_size] for i in range(0, len(text_data), row_size)]

            # Convert to DataFrame
            df = pd.DataFrame(rows)

            # Show table in Streamlit
            st.subheader("Extracted Table")
            st.dataframe(df)

            # Download as Excel
            excel_path = "extracted_table.xlsx"
            df.to_excel(excel_path, index=False)

            with open(excel_path, "rb") as f:
                st.download_button("Download Excel", f, file_name="extracted_table.xlsx")
