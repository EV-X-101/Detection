import cv2
import torch
import numpy as np

# Load a pre-trained YOLOv5 model (e.g., YOLOv5s)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load a pre-trained depth estimation model (MiDaS v2.1 Large)
midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_v2.1_large', pretrained=True)

# Set the device for running the models (CPU by default)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
midas_model.to(device)

# Open a connection to the camera (camera index 0 by default)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop to continuously capture frames from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection on the captured frame
    results = model(frame)
    detected_info = results.xyxy[0].cpu().numpy()

    # Display detected information
    for item in detected_info:
        x_min, y_min, x_max, y_max, confidence, class_idx = item
        class_name = results.names[int(class_idx)]
        print(f"Class: {class_name}, Confidence: {confidence:.2f}")
        print(f"Bounding Box: [{x_min}, {y_min}, {x_max}, {y_max}]")

        # Extract the region of interest (ROI) around the object
        roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

        # Estimate the distance to the object using the depth estimation model
        depth = midas_model(roi).mean()
        distance = 1 / depth

        # Print the estimated distance to the object
        print(f"Estimated Distance: {distance:.2f} meters")

        # Render the distance information on the frame
        cv2.putText(frame, f"Distance: {distance:.2f} m", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Render the results on the frame
    frame_with_results = results.render()[0]

    # Display the frame with detected objects
    cv2.imshow('YOLOv5 Object Detection', frame_with_results)

    # Press 'q' to exit the loop and close the camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
