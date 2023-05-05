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

# Define color range for red traffic light
red_lower_range1 = np.array([0, 50, 50])
red_upper_range1 = np.array([10, 255, 255])
red_lower_range2 = np.array([170, 50, 50])
red_upper_range2 = np.array([180, 255, 255])

# Define color range for green traffic light
green_lower_range = np.array([50, 100, 100])
green_upper_range = np.array([70, 255, 255])

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
    threat_distances = {'person': 0.5, 'car': 3.0, 'truck': 5.0, 'bus': 8.0, 'chair': 0.6} # Replace with the distances you want to detect each class at
    obstacle_detected = False
    traffic_light_detected = False
    
    # Detect traffic lights and determine their color
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_lower_range = np.array([0, 100, 100])
    red_upper_range = np.array([10, 255, 255])
    red_mask1 = cv2.inRange(hsv_frame, red_lower_range, red_upper_range)
    red_lower_range = np.array([160, 100, 100])
    red_upper_range = np.array([179, 255, 255])
    red_mask2 = cv2.inRange(hsv_frame, red_lower_range, red_upper_range)
    red_mask = cv2.addWeighted(red_mask1, 1.0, red_mask2, 1.0, 0.0)

    green_lower_range = np.array([50, 100, 100])
    green_upper_range = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv_frame, green_lower_range, green_upper_range)

    # Find contours in the red mask and determine if a red light is detected
    red_contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    red_light_detected = False
    for contour in red_contours:
        area = cv2.contourArea(contour)
        if area > 500:
            red_light_detected = True
            break

    # Find contours in the green mask and determine if a green light is detected
    green_contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_light_detected = False
    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area > 500:
            green_light_detected = True
            break

    # Send signal to Raspberry Pi to stop the car if a red light is detected
    if red_light_detected:
        print("Red light detected. Stopping car.")
        sock.sendall(b'stop')

    # Send signal to Raspberry Pi to move the car forward if no obstacle or traffic light is detected
    elif not obstacle_detected and not red_light_detected and not green_light_detected:
        print("No obstacle or traffic light detected. Moving car forward.")
        sock.sendall(b'forward')

    # Send signal to Raspberry Pi to move the car forward if a green light is detected
    elif green_light_detected:
        print("Green light detected. Moving car forward.")
        sock.sendall(b'forward')
    else:
        # Send signal to Raspberry Pi to stop the car if a red light or obstacle is detected
        if obstacle_detected:
            print("Obstacle detected. Stopping car.")
        else:
            print("Red light detected. Stopping car.")
        sock.sendall(b'stop')

    # Display the frame with detected objects and traffic lights
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Press 'q' to exit the loop and close the camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
