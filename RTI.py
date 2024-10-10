import cv2
import dlib

# Load face detector and recognizer
detector = dlib.get_frontal_face_detector()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:/Users/PRIYANKA/Desktop/Zecure/FaceRecognition/trainer/trainer.yml")
# Start video capture
cap = cv2.VideoCapture(0)
id = 0  # Sample user ID

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        if confidence < 50:
            cv2.putText(frame, f"User: {id}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Unknown", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
