from flask import Flask, jsonify, request # type: ignore
import cv2
from flask_cors import CORS  
import dlib
import numpy as np
import os
import threading
import time
from PIL import Image

app = Flask(__name__)
CORS(app)
# Global variables for face recognition
detector = dlib.get_frontal_face_detector()
recognizer = cv2.face.LBPHFaceRecognizer_create()
trainer_path = 'C:/Users/PRIYANKA/Desktop/Zecure/FaceRecognition/trainer/trainer.yml'
recognizer.read(trainer_path)
cap = cv2.VideoCapture(0)
detection_active = False
last_seen_time = time.time()
keypad_enabled = False

def process_frame():
    global detection_active, last_seen_time, keypad_enabled

    while True:
        if detection_active:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = detector(gray)
            detected_face = None

            if len(faces) > 0:
                detected_face = max(faces, key=lambda rect: rect.width() * rect.height())
                (x, y, w, h) = (detected_face.left(), detected_face.top(), detected_face.width(), detected_face.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                face_img = gray[y:y + h, x:x + w]
                id, confidence = recognizer.predict(face_img)
                if confidence < 50:
                    cv2.putText(frame, f"User: {id}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    keypad_enabled = True
                    last_seen_time = time.time()
                else:
                    cv2.putText(frame, "Unknown", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    keypad_enabled = False
            else:
                keypad_enabled = False

            if len(faces) == 0 or (keypad_enabled and time.time() - last_seen_time > 10):
                keypad_enabled = False

            if keypad_enabled:
                cv2.putText(frame, "Keypad Enabled", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Keypad Disabled", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

@app.route('/start', methods=['GET'])
def start_detection():
    global detection_active
    detection_active = True
    return jsonify({'status': 'Detection started'})

@app.route('/stop', methods=['GET'])
def stop_detection():
    global detection_active
    detection_active = False
    return jsonify({'status': 'Detection stopped'})

@app.route('/detect', methods=['POST'])
def detect_face():
    global keypad_enabled
    image_file = request.files['image']
    img = Image.open(image_file).convert('RGB')
    img_np = np.array(img)
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector(gray)
    results = []

    for face in faces:
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        face_img = gray[y:y + h, x:x + w]
        id, confidence = recognizer.predict(face_img)
        if confidence < 50:
            results.append({'x': x, 'y': y, 'w': w, 'h': h, 'id': id, 'confidence': confidence})
            keypad_enabled = True
        else:
            results.append({'x': x, 'y': y, 'w': w, 'h': h, 'id': 'Unknown', 'confidence': confidence})

    return jsonify(results)

@app.route('/train', methods=['GET'])
def train_model():
    dataset_path = 'C:/Users/PRIYANKA/Desktop/Zecure/MachineLearning/dataset'
    trainer_path = 'C:/Users/PRIYANKA/Desktop/Zecure/MachineLearning/trainer/trainer.yml'
    detector = cv2.CascadeClassifier("C:/Users/PRIYANKA/Desktop/Zecure/MachineLearning/haarcascade_frontalface_default.xml")
    if detector.empty():
        return jsonify({'status': 'Error loading Haar Cascade classifier.'})

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    def get_images_and_labels(path):
        face_samples = []
        ids = []
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_path in image_paths:
            img = Image.open(image_path).convert('L')  # Convert it to grayscale
            img_np = np.array(img, 'uint8')
            filename = os.path.split(image_path)[-1]
            try:
                id = int(filename.split('.')[0].replace('csv', ''))
            except ValueError:
                id = 1
            faces = detector.detectMultiScale(img_np)
            for (x, y, w, h) in faces:
                face_img = img_np[y:y + h, x:x + w]
                face_samples.append(face_img)
                ids.append(id)
        return face_samples, ids

    faces, ids = get_images_and_labels(dataset_path)
    if len(faces) > 0 and len(ids) > 0:
        recognizer.train(faces, np.array(ids))
        recognizer.save(trainer_path)
        return jsonify({'status': f'Model trained on {len(set(ids))} face(s)'})
    else:
        return jsonify({'status': 'No faces were detected or insufficient data for training.'})

if __name__ == '__main__':
    threading.Thread(target=process_frame).start()
    app.run(debug=True, use_reloader=False)
