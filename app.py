from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
import json
from keras.models import model_from_json
import time
import base64

app = Flask(__name__)

# Load model and Haar Cascade classifier
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
playlists = {
    'angry': 'https://youtube.com/playlist?list=PLe9Ra07TFcfnnzCSxE3nVU2P4uIPT7tiL&si=LB6UREpLHJeatnKm',
    'disgust': 'https://youtube.com/playlist?list=PLe9Ra07TFcfnnzCSxE3nVU2P4uIPT7tiL&si=LB6UREpLHJeatnKm',
    'fear': 'https://youtube.com/playlist?list=PLe9Ra07TFcfnnzCSxE3nVU2P4uIPT7tiL&si=LB6UREpLHJeatnKm',
    'happy': 'https://youtube.com/playlist?list=PLe9Ra07TFcfmo6i2v232bOF4C-1eeUySK&si=XoC5QjKOex0qLZNx',
    'neutral': 'https://youtube.com/playlist?list=PLe9Ra07TFcfnnzCSxE3nVU2P4uIPT7tiL&si=LB6UREpLHJeatnKm',
    'sad': 'https://youtube.com/playlist?list=PLe9Ra07TFcfmlP1aPKAGW1zgXXtHSYaK4&si=NIEzcV4nOO7jU2sS',
    'surprise': 'https://youtube.com/playlist?list=PLe9Ra07TFcfmo6i2v232bOF4C-1eeUySK&si=XoC5QjKOex0qLZNx'
}

def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotions = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        features = extract_features(face_roi)
        pred = model.predict(features)
        emotion_label = labels[pred.argmax()]
        emotions.append(emotion_label)

        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if emotions:
        detected_emotion = max(set(emotions), key=emotions.count)

    else:
        detected_emotion = None

    return image, detected_emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_from_image():
    if 'image' not in request.files:
        return render_template('index.html')

    image = request.files['image'].read()
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    detected_emotion, label = detect_emotion(img)
    _, buffer = cv2.imencode('.jpg', detected_emotion)
    img_base64 = base64.b64encode(buffer.tobytes()).decode()

    return render_template('index.html', detected_emotion=label, img_base64=img_base64, playlist=playlists[label])





@app.route('/realtime_emotion_detection')
def realtime_emotion_detection():
    return render_template('realtime_emotion_detection.html')

def detect_emotion_and_generate_frames():
    global model
    webcam = cv2.VideoCapture(0)
    start_time = time.time()
    emotion_counts = {label: 0 for label in labels.values()}
    global most_detected_emotion 

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
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    # Calculate the most detected emotion
    most_detected_emotion = max(emotion_counts, key=emotion_counts.get)
    print("Most detected emotion:", most_detected_emotion)
    
    
    # Release webcam
    webcam.release()
   
    
@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion_and_generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/suggest_music')
def suggest_music():
    playlists = {
        'angry': 'https://youtube.com/playlist?list=PLe9Ra07TFcfnnzCSxE3nVU2P4uIPT7tiL&si=LB6UREpLHJeatnKm',
     'disgust': 'https://youtube.com/playlist?list=PLe9Ra07TFcfnnzCSxE3nVU2P4uIPT7tiL&si=LB6UREpLHJeatnKm',
    'fear': 'https://youtube.com/playlist?list=PLe9Ra07TFcfnnzCSxE3nVU2P4uIPT7tiL&si=LB6UREpLHJeatnKm',
    'happy': 'https://youtube.com/playlist?list=PLe9Ra07TFcfmo6i2v232bOF4C-1eeUySK&si=XoC5QjKOex0qLZNx',
    'neutral': 'https://youtube.com/playlist?list=PLe9Ra07TFcfnnzCSxE3nVU2P4uIPT7tiL&si=LB6UREpLHJeatnKm',
    'sad': 'https://youtube.com/playlist?list=PLe9Ra07TFcfmlP1aPKAGW1zgXXtHSYaK4&si=NIEzcV4nOO7jU2sS',
    'surprise': 'https://youtube.com/playlist?list=PLe9Ra07TFcfmo6i2v232bOF4C-1eeUySK&si=XoC5QjKOex0qLZNx'
}
    
    global most_detected_emotion
    
    return jsonify(detected_emotion=most_detected_emotion, suggested_playlist=playlists[most_detected_emotion])
    
    
if __name__ == '__main__':
    app.run(debug=True)