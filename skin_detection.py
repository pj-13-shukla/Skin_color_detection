import cv2
import numpy as np
from ultralytics import YOLO

def detect_skin(face_region):
    # Convert the face region to the HSV color space
    hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

    # Define a range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask to extract skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Bitwise-AND the original image and the mask
    result = cv2.bitwise_and(face_region, face_region, mask=mask)

    return result

# Load the YOLOv8n-face model
model = YOLO("yolov8n-face.pt")

# Capture a frame from the video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform face detection using YOLOv8n-face
    results = model.predict(frame, conf=0.5)

    # Process each detected face
    for box in results[0].boxes:
        (x1, y1, x2, y2) = box.xyxy[0]

        # Draw rectangle around detected face
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Extract face region
        face_region = frame[int(y1):int(y2), int(x1):int(x2)]

        # Perform skin detection
        skin_image = detect_skin(face_region)

        # Display the skin detection result
        cv2.imshow('Detected Skin', skin_image)

    # Display the processed frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
