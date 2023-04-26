import cv2
import numpy as np
import torch

# Load a pre-trained YOLOv5 model that includes object classes
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

# Define baseline (distance between cameras) and focal length
b = 0.1 # meters
f = K[0, 0] # pixels

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

    # Display detected information and estimated distance
    for item in detected_info:
        x_min, y_min, x_max, y_max, confidence, class_idx = item
        class_name = results.names[int(class_idx)]
        print(f"Class: {class_name}, Confidence: {confidence:.2f}")
        print(f"Bounding Box: [{x_min}, {y_min}, {x_max}, {y_max}]")
        
        # Estimate the distance using stereo vision formula
        xl = x_min
        yl = y_min
        xr = x_max
        yr = y_max
        d = abs(xl - xr)
        depth = (b * f) / d
        print(f"Stereo Distance: {depth:.2f} meters")

        # Extract the region of interest (ROI) around the object
        roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

        # Estimate the distance to the object using the depth estimation model
        midas_depth = midas_model(roi).mean()
        midas_distance = 1 / midas_depth
        print(f"MiDaS Distance: {midas_distance:.2f} meters")

        # Render the distance information on the frame
        cv2.putText(frame, f"Class: {class_name}, Confidence: {confidence:.2f}",
                    (int(x_min), int(y_max) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"Stereo Distance: {depth:.2f} m",
                    (int(x_min), int(y_max) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"MiDaS Distance: {midas_distance:.2f} m",
                    (int(x_min), int(y_max) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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

