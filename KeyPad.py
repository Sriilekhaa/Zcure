import cv2
import dlib
import time

# Load face detector and recognizer
detector = dlib.get_frontal_face_detector()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:/Users/PRIYANKA/Desktop/Zecure/FaceRecognition/trainer/trainer.yml")

# Start video capture
cap = cv2.VideoCapture(0)
id = 0  # Sample user ID
last_seen_time = time.time()  # Time when user was last seen
keypad_enabled = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve detection
    gray = cv2.equalizeHist(gray)

    faces = detector(gray)
    detected_face = None

    # Process detected faces
    if len(faces) > 0:
        # Choose the largest face (most likely to be the primary face)
        detected_face = max(faces, key=lambda rect: rect.width() * rect.height())
        (x, y, w, h) = (detected_face.left(), detected_face.top(), detected_face.width(), detected_face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_img = gray[y:y + h, x:x + w]

        # Recognize the face
        id, confidence = recognizer.predict(face_img)
        print(f"ID: {id}, Confidence: {confidence}")

        if confidence < 50:
            cv2.putText(frame, f"User: {id}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            keypad_enabled = True
            last_seen_time = time.time()
        else:
            cv2.putText(frame, "Unknown", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            keypad_enabled = False
    else:
        keypad_enabled = False

    # Disable keypad if no face is detected or if the user has been away for more than 10 seconds
    if len(faces) == 0 or (keypad_enabled and time.time() - last_seen_time > 10):
        keypad_enabled = False

    # Display the keypad status
    if keypad_enabled:
        cv2.putText(frame, "Keypad Enabled", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Keypad Disabled", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
