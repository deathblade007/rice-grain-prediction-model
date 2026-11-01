import streamlit as st
import requests

st.title("Image Uploader & Predictor")

# Let user choose which classifier to run
model_choice = st.selectbox(
    "Select what you want to classify:",
    options=["Rice", "Flower"]
)

uploaded_file = st.file_uploader("Drag and drop your image here, or browse", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner(f"Running {model_choice} classification..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {"model_type": model_choice.lower()}

            try:
                response = requests.post("http://127.0.0.1:8000/predict", files=files, data=data)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction ({model_choice}): **{result['prediction']}**")
                else:
                    st.error(f"Error: Backend returned status code {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect. Make sure your FastAPI server is running in another terminal tab!")
