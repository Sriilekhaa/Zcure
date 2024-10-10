import cv2
import os
import numpy as np
from PIL import Image

# Correct path for saving the trained model and data
dataset_path = 'C:/Users/PRIYANKA/Desktop/Zecure/MachineLearning/dataset'
trainer_path = 'C:/Users/PRIYANKA/Desktop/Zecure/FaceRecognition/trainer/trainer.yml'

# Load the Haar Cascade classifier for face detection
detector = cv2.CascadeClassifier("C:/Users/PRIYANKA/Desktop/Zecure/MachineLearning/haarcascade_frontalface_default.xml")

if detector.empty():
    print("Error loading Haar Cascade classifier.")

# Create the recognizer object outside of the function
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to get images and labels for training
def get_images_and_labels(path):
    face_samples = []
    ids = []
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Image paths: {image_paths}")

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')  # Convert it to grayscale
        img_np = np.array(img, 'uint8')
        print(f"Image loaded: {image_path}, shape: {img_np.shape}")

        # Extract ID from the filename; default to 1 if not in expected format
        filename = os.path.split(image_path)[-1]
        try:
            id = int(filename.split('.')[0].replace('csv', ''))
        except ValueError:
            print(f"Filename format incorrect or ID extraction failed for: {filename}. Defaulting to ID 1.")
            id = 1

        faces = detector.detectMultiScale(img_np)

        if len(faces) == 0:
            print(f"No faces detected in image: {image_path}")

        for (x, y, w, h) in faces:
            face_img = img_np[y:y + h, x:x + w]
            face_samples.append(face_img)
            ids.append(id)
            print(f"Detected face with id {id}")

    return face_samples, ids

# Get face samples and ids
faces, ids = get_images_and_labels(dataset_path)
print(f"Number of faces detected: {len(faces)}")
print(f"Number of ids collected: {len(ids)}")

if len(faces) > 0 and len(ids) > 0:
    # Train the recognizer
    recognizer.train(faces, np.array(ids))

    # Save the trained model (ensure the path ends with the filename "trainer.yml")
    recognizer.save(trainer_path)
    print(f"Model trained on {len(set(ids))} face(s)")
else:
    print("No faces were detected or insufficient data for training.")
