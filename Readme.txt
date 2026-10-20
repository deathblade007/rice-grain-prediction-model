# Rice Grain Prediction Model

A machine learning project for classifying rice grain images into different rice varieties using a convolutional neural network (CNN) and a simple web interface.

This repository contains:
- a PyTorch CNN model for rice grain classification
- a FastAPI backend that serves predictions
- a Streamlit frontend for uploading and classifying images
- a class mapping file listing the supported rice varieties

## Overview

The project loads a trained model from `rice_model_weights.pth` and predicts the rice variety from a user-uploaded image. The backend processes the image, applies preprocessing transforms, runs inference, and returns the predicted class name. The frontend allows users to upload an image and trigger predictions through a browser UI.

## Model and Dataset

- Model type: CNN built in PyTorch
- Framework: PyTorch + torchvision
- Number of classes: 20
- Class list: defined in `class_mapping.json`
- Model weights file: `rice_model_weights.pth`

The model architecture includes multiple convolutional blocks, batch normalization, ReLU activations, max pooling, and a final classification head.

## Repository Structure

```text
.
├── app.py                  # Streamlit interface for uploading images
├── main.py                 # FastAPI backend and model inference logic
├── class_mapping.json      # Rice variety class names
├── rice_model_weights.pth  # Trained model weights
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md              # Project documentation