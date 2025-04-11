import streamlit as st
from detector import PhoneDetector
import os

st.title("Phone in Hand Detection")
st.write("Upload image(s) to detect if a phone is in hand.")

uploaded_files = st.file_uploader("Choose image(s)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    detector = PhoneDetector("yolov8m.pt")  # You can change the model path

    for file in uploaded_files:
        input_path = os.path.join("uploads", file.name)
        with open(input_path, "wb") as f:
            f.write(file.getbuffer())

        output_path, label = detector.detect_phones(input_path)

        st.image(output_path, caption=label, use_container_width=True)
