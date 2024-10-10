# Deep Learning Image Search Engine

This project is a content-based image retrieval (CBIR) system that leverages deep learning architectures (Inception V3, ResNet50, and VGG16) to extract image features and identify visually similar images. The system computes similarity scores based on extracted features and allows efficient searching and retrieval of similar images from a dataset.

## Overview
The goal of this project is to build an efficient image search engine using three popular convolutional neural network (CNN) models: Inception V3, ResNet50, and VGG16. The project extracts features from images using these models, reduces the feature dimensionality with PCA, and uses cosine similarity to find images most similar to a given query image.

## Features
- Feature extraction using deep learning models: Inception V3, ResNet50, VGG16
- Dimensionality reduction using Principal Component Analysis (PCA)
- Cosine similarity-based image similarity measurement
- Visualization of search results and feature embeddings using t-SNE

## Models Used
- **Inception V3**: Known for its efficient architecture and high performance on image classification tasks.
- **ResNet50**: A deep residual network that is widely used for its skip connections, making training of deep models easier.
- **VGG16**: A simple yet powerful model known for its uniform architecture and robust feature extraction capabilities.

## Results Visualization
- The top similar images to the query are displayed in a grid format.
- t-SNE is used to visualize how image features cluster together, showing the relationships between images in a low-dimensional space.

## Technologies
- Python 3
- TensorFlow / Keras for deep learning
- scikit-learn for PCA and distance calculations
- Matplotlib and Seaborn for visualization
- NumPy for numerical operations
