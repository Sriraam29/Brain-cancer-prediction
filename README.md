# Brain-cancer-prediction
Brain cancer prediction using CNN model
This project uses a Convolutional Neural Network (CNN) to predict the presence of brain tumors in MRI scans. The model is designed to classify MRI images as either having a brain tumor or being tumor-free, which can assist in early diagnosis and treatment planning.
Overview
This repository contains code for training and evaluating a CNN model that classifies MRI images into two classes:

Tumor
No Tumor
The model is based on a simple yet effective CNN architecture, which has been tuned to provide high accuracy on this task. The model is implemented in Python using TensorFlow and Keras.
Dataset
The dataset used for training consists of MRI images labeled as either "tumor" or "no tumor." Each image is preprocessed and resized to a standard input size before being fed into the CNN. You can download the dataset from Kaggle's Brain MRI Dataset or another source with similar labeled images.

Data Preprocessing
Each image is resized and normalized before feeding it to the model. Data augmentation techniques (such as rotation, zooming, and flipping) are applied to increase dataset variability and improve model generalization.
Model Architecture
The CNN model is designed with the following architecture:

Input Layer: Accepts a preprocessed MRI image.
Convolutional Layers: Extracts features from images using several convolution and max-pooling layers.
Fully Connected Layers: Combines the features and prepares them for classification.
Output Layer: Uses a sigmoid function for binary classification (tumor/no tumor).
The model is compiled with binary cross-entropy loss and the Adam optimizer.
Installation
Prerequisites
To run this project, you'll need the following libraries:

Python 3.x
TensorFlow
Keras
NumPy
Matplotlib
Scikit-learn
OpenCV (optional, for image processing)
