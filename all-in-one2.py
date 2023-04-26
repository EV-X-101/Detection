import cv2
import numpy as np
import torch

# Load a pre-trained YOLOv5 model that includes traffic lights and other objects
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)

# Load a pre-trained depth estimation model (MiDaS v2.1 Large)
midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_v2.1_large', pretrained=True)

# Define camera calibration matrix and distortion coefficients
K = np.array([[5.9421434211923245e+02, 0., 3.1950000000000000e+02],
              [0., 5.9421434211923245e+02, 2.3950000000000000e+02],
              [0., 0., 1.]])
D = np.array([-4.0193755660864893e-01, 2.4463675579426594e-01,
              -4.2951404082820734e-04, -1.7035632257473457e-04,
              -6.8671397083610292e-02])

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

    # Display detected information and estimated distance for objects and traffic lights
    for item in detected_info:
        x_min, y_min, x_max, y_max, confidence, class_idx = item
        class_name = results.names[int(class_idx)]

        if class_name == 'person':
            # Render a rectangle around the detected person
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        elif class_name != 'traffic light':
            # Render a rectangle around the detected object
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

            # Extract the region of interest (ROI) around the object
            roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

            # Estimate the distance to the object using the depth estimation model
            midas_depth = midas_model(roi).mean()
            midas_distance = 1 / midas_depth
            print(f"{class_name}: {midas_distance:.2f} meters")

            # Render the distance information on the frame
            cv2.putText(frame, f"{class_name}: {midas_distance:.2f} m",
                        (int(x_min), int(y_max) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        else:
            # Render the traffic light information on the frame
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}",
                        (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Render the results on the frame
    frame_with_results = results.render()[0]

    # Display the frame with detected objects and traffic lights
    cv2.imshow('YOLOv5 Object and Traffic Light Detection', frame_with_results)

    # Press 'q' to exit the loop and close the camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
