# import cv2
# from ultralytics import YOLO

# # Load the YOLOv8n-face model
# model = YOLO("yolov8n-face.pt")

# # Capture a frame from the video
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform face detection using YOLOv8n-face
#     results = model.predict(frame, conf=0.5)

#     # Draw rectangles around detected faces
#     for box in results[0].boxes:
#         (x1, y1, x2, y2) = box.xyxy[0]
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

#     cv2.imshow('frame', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from ultralytics import YOLO

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

        # Evaluate lighting (example: calculate average brightness)
        avg_brightness = np.mean(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))

        # Evaluate face position (example: check if face is centered horizontally)
        face_center_x = (x1 + x2) / 2
        image_center_x = frame.shape[1] / 2
        position_good = abs(face_center_x - image_center_x) < frame.shape[1] * 0.1  # Adjust the threshold as needed

        # Evaluate if face is looking straight (example: check if eyes are roughly at the same horizontal level)
        eyes_center_y = (y1 + y2) / 2
        position_straight = abs(y1 - eyes_center_y) < frame.shape[0] * 0.1  # Adjust the threshold as needed

        # Measure face distance from the camera (example: use width of the face bounding box)
        face_distance = x2 - x1

        # Display evaluation results on the frame
        cv2.putText(frame, f'Avg. Brightness: {avg_brightness:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f'Face Position: {"Good" if position_good else "Bad"}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f'Looking Straight: {"Yes" if position_straight else "No"}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f'Face Distance: {face_distance:.2f} pixels', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the processed frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

