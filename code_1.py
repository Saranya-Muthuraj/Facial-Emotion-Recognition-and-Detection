import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Load FER-2013 Dataset
fer_df = pd.read_csv('fer2013.csv')  # Make sure this CSV is in your working directory

# Filter out rows with incorrect pixel length to avoid errors
fer_df = fer_df[fer_df['pixels'].apply(lambda x: len(x.split()) == 48*48)].reset_index(drop=True)

# Prepare the pixel data
X = np.array([np.fromstring(pixels, dtype=np.float32, sep=' ') for pixels in fer_df['pixels']])
X = X.reshape(-1, 48, 48, 1) / 255.0  # Normalize pixel values

# Encode the emotions as one-hot vectors
y = to_categorical(fer_df['emotion'], num_classes=7)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model architecture
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(2,2),
        
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 classes for emotions
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model_path = 'emotion_model.h5'

if os.path.exists(model_path):
    print("Loading existing trained model...")
    model = load_model(model_path)
else:
    print("Training new model...")
    model = create_model()
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        callbacks=[checkpoint, early_stop]
    )
    
    # Plot accuracy and loss graphs (non-blocking)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    # Evaluate on test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")

# Function to find working camera index
def find_working_camera_index(max_index=5):
    for index in range(max_index):
        temp_cap = cv2.VideoCapture(index)
        if temp_cap.read()[0]:
            temp_cap.release()
            return index
        temp_cap.release()
    return -1

# Real-Time Emotion Detection using webcam
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

camera_index = find_working_camera_index()
if camera_index == -1:
    print("No working camera found. Exiting...")
    exit()
else:
    print(f"Using camera index: {camera_index}")

cap = cv2.VideoCapture(camera_index)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Starting real-time emotion detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (48,48))
        normalized_face = resized_face / 255.0
        reshaped_face = normalized_face.reshape(1,48,48,1)
        
        prediction = model.predict(reshaped_face)
        emotion = emotion_labels[np.argmax(prediction)]
        
        # Draw rectangle and put emotion text
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    
    cv2.imshow('Facial Emotion Recognition', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("Exiting real-time detection.")
        break

cap.release()
cv2.destroyAllWindows()
