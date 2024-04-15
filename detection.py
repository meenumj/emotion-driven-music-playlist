import cv2
import numpy as np
from keras.models import model_from_json
import time

# Load model and Haar Cascade classifier

# Load model architecture
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()

# Load model weights
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar Cascade classifier for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define function to extract features
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

# Access webcam
webcam = cv2.VideoCapture(0)

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Initialize dictionary to store counts of each emotion
emotion_counts = {label: 0 for label in labels.values()}

# Capture frames for 15 seconds
start_time = time.time()
try:
    while (time.time() - start_time) < 5:
        ret, frame = webcam.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]

            # Resize face region to match model input size
            face_roi = cv2.resize(face_roi, (48, 48))

            # Extract features and normalize
            features = extract_features(face_roi)

            # Predict emotion
            pred = model.predict(features)
            emotion_label = labels[pred.argmax()]

            # Update count for detected emotion
            emotion_counts[emotion_label] += 1

            # Display emotion label on the frame
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display frame
        cv2.imshow("Emotion Detection", frame)
        cv2.waitKey(1)

except KeyboardInterrupt:
    pass

# Calculate the most detected emotion
most_detected_emotion = max(emotion_counts, key=emotion_counts.get)
print("Most detected emotion:", most_detected_emotion)

# Release webcam and close windows
webcam.release()
cv2.destroyAllWindows()
