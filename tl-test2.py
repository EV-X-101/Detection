import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box


# Load the pre-trained YOLOv5 model for traffic sign and traffic light detection
model_path = 'models/yolov5s_traffic.pt'
model = attempt_load(model_path, map_location=torch.device('cpu'))
model.eval()

# Define the class names for the traffic signs and traffic lights that the model can detect
class_names = ['traffic_light', 'stop_sign', 'yield_sign', 'speed_limit_sign']

# Define the color ranges for the traffic lights in HSV color space
red_lower_range1 = np.array([0, 70, 50])
red_upper_range1 = np.array([10, 255, 255])

red_lower_range2 = np.array([160, 70, 50])
red_upper_range2 = np.array([180, 255, 255])

yellow_lower_range = np.array([20, 100, 100])
yellow_upper_range = np.array([30, 255, 255])

green_lower_range = np.array([40, 70, 70])
green_upper_range = np.array([90, 255, 255])

# Open a connection to the camera (camera index 0 by default)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop over the frames captured from the camera and perform traffic sign and traffic light detection on each frame
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize and pad the frame to the input size of the YOLOv5 model
    img_size = 640
    img = letterbox(frame, new_shape=img_size)[0]

    # Convert the frame to RGB color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PyTorch tensor
    img = torch.from_numpy(img).to(torch.device('cpu'))
    img = img.permute(2, 0, 1).unsqueeze(0).float()

    # Perform object detection on the frame using the YOLOv5 model
    results = model(img)

    # Get the detected object information
    detected_info = results.xyxy[0].cpu().numpy()

    # Loop over the detected objects and perform traffic sign and traffic light detection on each object
    for item in detected_info:
        x_min, y_min, x_max, y_max, confidence, class_idx = item
        class_name = class_names[int(class_idx)]

        # Extract the region of interest (ROI) around the object
        roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

        # Check if the object is a traffic light
        if class_name == 'traffic_light':
            # Extract the region of interest (ROI) for the traffic light
            traffic_light_roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

            # Convert the ROI to HSV color space
            hsv_roi = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)
            
            # Threshold the image to extract the color of the traffic light
            mask1 = cv2.inRange(hsv_roi, red_lower_range1, red_upper_range1)
            mask2 = cv2.inRange(hsv_roi, red_lower_range2, red_upper_range2)
            red_mask = mask1 + mask2

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
                traffic_light_color = 'unknown'

            # Render the traffic light color on the frame
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
            cv2.putText(frame, f"Traffic light: {traffic_light_color}", (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check if the object is a traffic sign
        elif class_name in ['stop_sign', 'yield_sign', 'speed_limit_sign']:
            # Render the traffic sign label on the frame
            label = class_name.replace('_', ' ').capitalize()
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Traffic sign and traffic light detection', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
