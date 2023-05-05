import cv2
import numpy as np
import torch
import socket
import os

# Get the IP address from the environment variable
RPI_IP_ADDRESS = os.environ.get('RPI_IP_ADDRESS')
# Load a pre-trained YOLOv5 model (e.g., YOLOv5s)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load a pre-trained depth estimation model (MiDaS v2.1 Large)
midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_v2.1_large', pretrained=True)

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

def traffic_light_state(frame, bbox):
    xmin, ymin, xmax, ymax = [int(coord) for coord in bbox]
    cropped_frame = frame[ymin:ymax, xmin:xmax]

    if cropped_frame.size == 0:
        print("Empty cropped frame!")
        return "Unknown"

    # Convert the cropped image to BGR and then to HSV color space
    bgr_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
    red_lower_range1 = np.array([0, 70, 50])
    red_upper_range1 = np.array([10, 255, 255])

    red_lower_range2 = np.array([160, 70, 50])
    red_upper_range2 = np.array([180, 255, 255])

    yellow_lower_range = np.array([20, 100, 100])
    yellow_upper_range = np.array([30, 255, 255])

    green_lower_range = np.array([40, 70, 70])
    green_upper_range = np.array([90, 255, 255])

    # Create masks for each color range
    red_mask1 = cv2.inRange(hsv_frame, red_lower_range1, red_upper_range1)
    red_mask2 = cv2.inRange(hsv_frame, red_lower_range2, red_upper_range2)
    red_mask = cv2.addWeighted(red_mask1, 1.0, red_mask2, 1.0, 0.0)

    yellow_mask = cv2.inRange(hsv_frame, yellow_lower_range, yellow_upper_range)
    green_mask = cv2.inRange(hsv_frame, green_lower_range, green_upper_range)

    # Count the number of pixels for each color
    red_count = cv2.countNonZero(red_mask)
    yellow_count = cv2.countNonZero(yellow_mask)
    green_count = cv2.countNonZero(green_mask)

    # If yellow pixels are detected, prioritize yellow color
    if yellow_count > 10:
        return "Yellow"

    # Determine the traffic light state based on the number of colored pixels
    color_counts = [("Red", red_count), ("Green", green_count)]
    color_counts.sort(key=lambda x: x[1], reverse=True)
    return color_counts[0][0]


# Open a connection to the camera (camera index 0 by default)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set up socket connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((RPI_IP_ADDRESS, 5000)) # Replace with Raspberry Pi IP address
print('\b\b\b--- Socket connection established ----')

# Loop to continuously capture frames from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection on the captured frame
    results = model(frame)
    detected_info = results.xyxy[0].cpu().numpy()

    # Check if an obstacle is detected and send signal to Raspberry Pi
    threat_classes = ['person', 'car', 'truck', 'bus', 'chair'] # Replace with the classes you want to detect
    threat_distances = {'person': 1.0, 'car': 3.0, 'truck': 5.0, 'bus': 8.0, 'chair': 0.8} # Replace with the distances you want to detect each class at
    obstacle_detected = False
    for item in detected_info:
        x_min, y_min, x_max, y_max, confidence, class_idx = item
        class_name = results.names[int(class_idx)]
        
        # Extract the region of interest (ROI) around the object
        roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

        # Estimate the distance to the object using the depth estimation model
        depth = midas_model(roi).mean()
        distance = 1 / depth

        # Display the object name and estimated distance
        text = f"{class_name} {distance:.2f}m"
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
        cv2.putText(frame, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        print(f"Detected: {text}")

        # Check if an obstacle is detected and send signal to Raspberry Pi
        for threat_class in threat_classes:
            if class_name == threat_class and distance < threat_distances[threat_class]:
                obstacle_detected = True
                print(f"{threat_class.capitalize()} detected. Stopping car.")
                sock.sendall(b'stop')
                break

        # Check if the object is a traffic light
        if class_name == 'traffic light':
            # Extract the region of interest (ROI) for the traffic light
            traffic_light_roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

            # Get the state of the traffic light
            traffic_light_color = traffic_light_state(frame, (x_min, y_min, x_max, y_max))

            # Render the traffic light color on the frame
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
            cv2.putText(frame, f"Color: {traffic_light_color}", (int(x_min), int(y_min) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Check if the traffic light is red or yellow and send signal to Raspberry Pi to stop the car
            if traffic_light_color == 'red' or traffic_light_color == 'yellow':
                obstacle_detected = True
                print(f"Traffic light is {traffic_light_color}. Stopping car.")
                sock.sendall(b'stop')

    # Send signal to Raspberry Pi to move car forward if no obstacle or red traffic light is detected
    if not obstacle_detected:
        print("No obstacle or red traffic light detected. Moving car forward.")
        sock.sendall(b'forward')

    # Display the frame with detected objects
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Press 'q' to exit the loop and close the camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
