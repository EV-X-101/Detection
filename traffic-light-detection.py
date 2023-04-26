import cv2
import torch
import numpy as np

# Load a pre-trained YOLOv5 model (e.g., YOLOv5s)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

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

    # Perform inference on the captured frame
    results = model(frame)
    detected_info = results.xyxy[0].cpu().numpy()

    # Display detected information
    for item in detected_info:
        x_min, y_min, x_max, y_max, confidence, class_idx = item
        class_name = results.names[int(class_idx)]
        if class_name == 'traffic light':
            # Extract the region of interest (ROI) around the traffic light
            roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

            # Convert the ROI to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Define the color ranges for red, yellow, and green lights
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            lower_green = np.array([50, 100, 100])
            upper_green = np.array([70, 255, 255])

            # Apply color segmentation to the ROI to determine the color of the traffic light
            mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)
            mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
            mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
            red_pixels = cv2.countNonZero(mask_red)
            yellow_pixels = cv2.countNonZero(mask_yellow)
            green_pixels = cv2.countNonZero(mask_green)
            color = 'unknown'
            if red_pixels > yellow_pixels and red_pixels > green_pixels:
                color = 'red'
            elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
                color = 'yellow'
            elif green_pixels > red_pixels and green_pixels > yellow_pixels:
                color = 'green'
            
            print(f"Traffic Light: Color: {color}, Confidence: {confidence:.2f}")
            print(f"Bounding Box: [{x_min}, {y_min}, {x_max}, {y_max}]")
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

    # Display the frame with detected objects
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Press 'q' to exit the loop and close the camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the
