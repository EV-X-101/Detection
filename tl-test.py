import cv2
import torch
import numpy as np

# Load a pre-trained YOLOv5 model (e.g., YOLOv5s)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# Set the device for running the models (CPU by default)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Open a connection to the camera (camera index 0 by default)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define color ranges in HSV
red_lower_range1 = np.array([0, 70, 50])
red_upper_range1 = np.array([10, 255, 255])

red_lower_range2 = np.array([160, 70, 50])
red_upper_range2 = np.array([180, 255, 255])

yellow_lower_range = np.array([20, 100, 100])
yellow_upper_range = np.array([30, 255, 255])

green_lower_range = np.array([40, 70, 70])
green_upper_range = np.array([90, 255, 255])

# Set initial traffic light color to None
traffic_light_color = None
prev_traffic_light_color = None

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

        # Extract the region of interest (ROI) around the object
        roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

        # Render the distance information on the frame
        cv2.putText(frame, f"Distance:", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

        # Check if the object is a traffic light
        if class_name == 'traffic light':
            # Extract the region of interest (ROI) for the traffic light
            traffic_light_roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

            # Convert the ROI to HSV color space
            hsv_roi = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)

            # Threshold the image to extract the color of the traffic light
            mask1 = cv2.inRange(hsv_roi, red_lower_range1, red_upper_range1)
            mask2 = cv2.inRange(hsv_roi, red_lower_range2, red_upper_range2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            yellow_mask = cv2.inRange(hsv_roi, yellow_lower_range, yellow_upper_range)

            green_mask = cv2.inRange(hsv_roi, green_lower_range, green_upper_range)

            # Determine the color of the traffic light
            if cv2.countNonZero(red_mask) > 10:
                traffic_light_color = 'red'
            elif cv2.countNonZero(yellow_mask) > 10:
                traffic_light_color = 'yellow'
            elif cv2.countNonZero(green_mask) > 10:
                traffic_light_color = 'green'
            else:
                traffic_light_color = None

            # If the traffic light color has changed, print the new color
            if traffic_light_color is not None and traffic_light_color != prev_traffic_light_color:
                print(f"Traffic Light: {traffic_light_color}")
                prev_traffic_light_color = traffic_light_color

            # Render the traffic light color on the frame
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
            cv2.putText(frame, f"Color: {traffic_light_color}", (int(x_min), int(y_min) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

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
