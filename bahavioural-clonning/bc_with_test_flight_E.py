import cv2
import numpy as np
import torch
import socket
import os
from tensorflow.keras.models import load_model

# Load both models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bc_model = load_model('C:/Users/firao/Documents/Detection and Depth Estimation/Detection/bahavioural-clonning/bc.h5')
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

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

# Preprocessing and prediction for the behavioral cloning model
def bc_predict(image):
    # Preprocess image
    image = cv2.resize(image, (200, 66))  # resize to match the size the model expects
    image = image / 255.  # normalize pixel values
    image = np.expand_dims(image, axis=0)  # add an extra dimension for batch size
    
    # Pass the image through the model
    output = bc_model.predict(image)

    # Postprocess the output
    # As your model only returns the steering angle, we don't need to take argmax
    steering_angle = output[0][0]
    # Depending on the steering angle, we will decide the turn command
    if steering_angle < -0.1:
        turn_command = '2'  # left
    elif steering_angle > 0.1:
        turn_command = '1'  # right
    else:
        turn_command = '0'  # center

    return turn_command

# Traffic light state function is unchanged from your original code
def traffic_light_state(frame, bbox):
    # Code remains same as your original function
    ...

# Set up socket connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('RPI_ADDRESS', 5000)) # Replace with Raspberry Pi IP address
print('Socket connection established')

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

    # Perform inference on the captured frame using the YOLO model
    yolo_results = yolo_model(frame)
    detected_info = yolo_results.xyxy[0].cpu().numpy()

    # Also preprocess the frame and get the turning command from the behavioral cloning model
    turn_command = bc_predict(frame)

    # Similar to your original code, just added the integration with the behavioral cloning model
    threat_classes = ['person', 'car', 'truck', 'bus']
    threat_distances = {'person': 0.5, 'car': 3.0, 'truck': 5.0, 'bus': 8.0}
    obstacle_detected = False
    for item in detected_info:
        # Similar processing as your original code
        ...

        # If no obstacle is detected, use the turning command from the behavioral cloning model
        if not obstacle_detected:
            print("No obstacle detected. Moving car forward.")
            if turn_command == '0':  # center
                sock.sendall(b'forward')
            elif turn_command == '1':  # right
                sock.sendall(b'right')
            else:  # left
                sock.sendall(b'left')

    # Display the frame with detected objects
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Press 'q' to exit the loop and close the camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
