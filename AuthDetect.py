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
start_time = time.time()  # Start time for user turning phase
turn_stage = 0  # 0 = Initial, 1 = Turn 1, 2 = Turn 2, 3 = Success, 4 = Retry
turn_timeout = 120  # 2 minutes in seconds

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
            if not keypad_enabled:
                # Initialize the turning phase
                keypad_enabled = True
                turn_stage = 1
                start_time = time.time()
            elif turn_stage == 1:
                # Show instruction for first turn
                cv2.putText(frame, "Turn your face left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif turn_stage == 2:
                # Show instruction for second turn
                cv2.putText(frame, "Turn your face right", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif turn_stage == 3:
                # Sign-in successful message
                cv2.putText(frame, "Sign-in Successful", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                keypad_enabled = False
                turn_stage = 0
        else:
            # If confidence is low, reset turning phase
            if turn_stage > 0 and time.time() - start_time > turn_timeout:
                cv2.putText(frame, "Please try again", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                keypad_enabled = False
                turn_stage = 0
                start_time = time.time()
    else:
        keypad_enabled = False
        if turn_stage > 0 and time.time() - start_time > turn_timeout:
            cv2.putText(frame, "Please try again", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            turn_stage = 0
            start_time = time.time()

    # Handle the turn stages and timeout
    if keypad_enabled:
        if turn_stage == 1 and time.time() - start_time > turn_timeout / 2:
            # Transition from turn 1 to turn 2
            turn_stage = 2
            start_time = time.time()
        elif turn_stage == 2 and time.time() - start_time > turn_timeout:
            # Final check for turn 2 completion
            turn_stage = 3
            start_time = time.time()

    cv2.imshow("Face Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
