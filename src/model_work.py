import cv2
import joblib
import numpy as np

# Load the trained KNN model
model = joblib.load("ddd.pkl")

# Load Haar Cascade Classifier for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start video capture (webcam)
cap = cv2.VideoCapture(0)

# Counter for drowsiness detection
drowsy_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in eyes:
        eye = gray[y:y+h, x:x+w]
        eye_resized = cv2.resize(eye, (50, 50))  # Resizing to match training input
        eye_flatten = eye_resized.flatten()
        
        # Predicting using KNN model
        prediction = model.predict(eye_flatten.reshape(1,-1))

        if prediction[0] == 1:  # 1 = Drowsy
            drowsy_counter += 1
            label = "Drowsy"
            color = (0, 0, 255)
        else:
            drowsy_counter = 0
            label = "Awake"
            color = (0, 255, 0)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # If drowsiness detected for multiple frames
    if drowsy_counter >= 10:
        cv2.putText(frame, "ALERT! DRIVER DROWSY!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Driver Drowsiness Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and destroy all windows
cap.release()
cv2.destroyAllWindows()