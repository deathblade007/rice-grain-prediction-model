# Rice Grain Prediction Model

This repository presents a deep learning-based image classification system for identifying rice grain varieties from digital images. The project combines a custom convolutional neural network (CNN), a FastAPI-based inference service, and a Streamlit user interface to provide a complete end-to-end pipeline for automated rice grain classification.

## Abstract

Rice grain identification is a challenging visual recognition task due to subtle differences in shape, color, texture, and grain morphology across varieties. In this work, a convolutional neural network is trained to classify rice grain images into predefined categories. The system is implemented in Python using PyTorch and is deployed locally through a lightweight web interface.

The repository contains the trained model weights, the inference backend, and a frontend application for interactive prediction.

## Objectives

The primary objectives of this project are to:

- classify rice grain images into multiple known varieties,
- develop a lightweight CNN architecture suitable for image recognition tasks,
- provide an easy-to-use local application for inference,
- enable rapid prototyping and experimentation with computer vision pipelines.

## Methodology

### Model

The model is implemented as a CNN using PyTorch. It consists of multiple convolutional blocks with:

- convolutional layers,
- batch normalization,
- ReLU activation,
- max pooling,
- fully connected classification layers.

The final layer maps the extracted feature representation to one of the target classes defined in `class_mapping.json`.

### Data Preprocessing

Before inference, uploaded images are processed as follows:

- image is read from the request payload,
- converted to RGB format,
- resized to a fixed spatial dimension,
- transformed into a tensor,
- normalized using standard ImageNet-style normalization parameters.

### Inference Pipeline

The application performs the following sequence:

1. User uploads an image through the Streamlit interface.
2. The frontend sends the image to the FastAPI backend.
3. The backend preprocesses the image.
4. The trained model predicts the class index.
5. The predicted index is mapped to the corresponding rice variety name.
6. The result is returned to the frontend and displayed to the user.

## Repository Structure

```text
rice-grain-prediction-model/
├── app.py                  # Streamlit web application for user interaction
├── main.py                 # FastAPI backend and model inference pipeline
├── class_mapping.json      # Mapping between class indices and rice names
├── rice_model_weights.pth  # Pretrained model weights
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md               # Project documentation