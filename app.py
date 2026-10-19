import streamlit as st
import requests

st.title("Image Uploader & Predictor")

uploaded_file = st.file_uploader("Drag and drop your image here, or browse", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Predict"):
        with st.spinner("Processing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: {result['prediction']}")
                else:
                    st.error(f"Error: Backend returned status code {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect. Make sure your FastAPI server is running in another terminal tab!")
