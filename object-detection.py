import cv2
import numpy as np
import torch
import socket
import os

# Get the IP address from the environment variable
RPI_IP_ADDRESS = os.environ.get('RPI_IP_ADDRESS')
# Load a pre-trained YOLOv5 model (e.g., YOLOv5s)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define camera calibration matrix and distortion coefficients
K = np.array([[5.9421434211923245e+02, 0., 3.1950000000000000e+02],
              [0., 5.9421434211923245e+02, 2.3950000000000000e+02],
              [0., 0., 1.]])
D = np.array([-4.0193755660864893e-01, 2.4463675579426594e-01,
              -4.2951404082820734e-04, -1.7035632257473457e-04,
              -6.8671397083610292e-02])

# Define baseline (distance between cameras) and focal length
b = 0.1 # meters
f = K[0, 0] # pixels

# Open a connection to the camera (camera index 0 by default)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set up socket connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((RPI_IP_ADDRESS, 5000)) # Replace with Raspberry Pi IP address
print('Socket connection established')
# Loop to continuously capture frames from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the captured frame
    results = model(frame)
    detected_info = results.xyxy[0].cpu().numpy()

    # Check if an obstacle is detected and send signal to Raspberry Pi
    threat_classes = ['person', 'car', 'truck', 'bus', 'chair'] # Replace with the classes you want to detect
    threat_distances = {'person': 1.0, 'car': 3.0, 'truck': 5.0, 'bus': 8.0, 'chair': 0.8} # Replace with the distances you want to detect each class at
    obstacle_detected = False
    for item in detected_info:
        x_min, y_min, x_max, y_max, confidence, class_idx = item
        class_name = results.names[int(class_idx)]
        
        # Estimate the distance using stereo vision formula
        xl = x_min
        yl = y_min
        xr = x_max
        yr = y_max
        d = abs(xl - xr)
        depth = (b * f) / d
        
        # Display the object name and estimated depth
        text = f"{class_name} {depth:.2f}m"
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
        cv2.putText(frame, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        print(f"Detected: {text}")
        
        # Check if an obstacle is detected and send signal to Raspberry Pi
        for threat_class in threat_classes:
            if class_name == threat_class and depth < threat_distances[threat_class]:
                obstacle_detected = True
                print(f"{threat_class.capitalize()} detected. Stopping car.")
                sock.sendall(b'stop')
                break

    # Send signal to Raspberry Pi to move car forward if no obstacle is detected
    if not obstacle_detected:
        print("No obstacle detected. Moving car forward.")
        sock.sendall(b'forward')

    # Display the frame with detected objects
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Press 'q' to exit the loop and close the camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
