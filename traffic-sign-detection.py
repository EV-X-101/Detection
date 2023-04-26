import cv2
import numpy as np
import pytesseract
import torch

# Load a pre-trained YOLOv5 model (e.g., YOLOv5s)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the device for running the model (CPU by default)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the minimum confidence score for object detection
min_confidence = 0.5

# Define the Tesseract OCR engine configuration
tess_config = '--psm 11 --oem 3'

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

    # Display detected information and recognized text
    for item in detected_info:
        x_min, y_min, x_max, y_max, confidence, class_idx = item
        class_name = results.names[int(class_idx)]
        
        # Check if the detected object is a traffic sign
        if class_name.startswith('traffic sign') and confidence >= min_confidence:
            print(f"Traffic Sign: {class_name}, Confidence: {confidence:.2f}")
            print(f"Bounding Box: [{x_min}, {y_min}, {x_max}, {y_max}]")

            # Extract the region of interest (ROI) around the traffic sign
            roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

            # Convert the ROI to grayscale and apply thresholding
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Perform text recognition on the thresholded ROI using Tesseract
            text = pytesseract.image_to_string(thresh, config=tess_config)

            # Print the recognized text
            print(f"Recognized Text: {text}")

            # Render the recognized text on the frame
            cv2.putText(frame, f"Sign: {text}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Render the results on the frame
    frame_with_results = results.render()[0]

    # Display the frame with detected objects and recognized text
    cv2.imshow('Traffic Sign Detection', frame_with_results)

    # Press 'q' to exit the loop and close the camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
