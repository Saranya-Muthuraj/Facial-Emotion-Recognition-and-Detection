# Facial Emotion Recognition using CNN and OpenCV 🎭

This project implements a real-time **Facial Emotion Recognition System** using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset and integrated with OpenCV to detect and label emotions via webcam.

---

## 🔍 Overview

- Trains a CNN on FER-2013 dataset (`fer2013.csv`)
- Detects and classifies emotions from webcam in real time
- Emotions Supported: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`

---

## 📁 Dataset

Download the FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place `fer2013.csv` in the root directory of this project.

---

## 🧰 Requirements

Install required libraries:

```bash
pip install numpy pandas matplotlib opencv-python scikit-learn tensorflow
```
## 🚀 How to Run

## 1. Train or Load the Model

Run the Python script to start training or load an existing model:

```bash    Copy    Edit
python your_script_name.py
```
- If emotion_model.h5 exists, it will be loaded automatically.

- Otherwise, the CNN model will be trained and saved.

## 2. Real-Time Emotion Detection

Once training/loading completes, your webcam will activate and display a live feed with detected faces labeled with predicted emotions.

Press ``q`` to exit the real-time detection window.

## 🧠 Model Architecture

- ``Conv2D Layers``: 3 layers with increasing filters (32 → 64 → 128)

- ``MaxPooling2D``: After each Conv2D layer

- ``Dropout``: For regularization (0.25 and 0.5)

- ``Dense``: 128 neurons with ReLU, followed by 7-output softmax

## 💡 Features

- Real-time face detection using OpenCV

- Emotion prediction using CNN with TensorFlow/Keras

- Model auto-saves and reloads for reusability

- Accuracy and loss plots during training

## 📱 Future Scope

- Convert model to TensorFlow Lite for mobile deployment

- Build Android/iOS app using Flutter or React Native

- Integrate with Raspberry Pi or edge devices for on-device recognition

## 📜 License

This project is licensed under the MIT License.

## 🙋 Author

Saranya M

[GitHub Profile]

https://github.com/Saranya-Muthuraj

