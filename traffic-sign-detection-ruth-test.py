import cv2
import numpy as np
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define the classes of traffic signs
classes = ['stop', 'yield', 'speed_limit', 'turn_left', 'turn_right']

# Define the function to detect traffic signs
def detect_traffic_signs(image):
  # Convert the image to a tensor
  image_tensor = torch.from_numpy(image).float()

  # Detect the objects in the image
  detections = model(image_tensor)

  # Loop over the detections
  for detection in detections:
    # Get the class of the object
    class_id = detection['class_id']

    # Get the confidence of the detection
    confidence = detection['confidence']

    # Check if the confidence is above a certain threshold
    if confidence > 0.5:
      # The confidence is above a certain threshold, so the object is detected

      # Get the bounding box of the object
      bounding_box = detection['bbox']

      # Draw a rectangle around the object
      cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 255, 0), 2)

      # Display the object name on the image
      cv2.putText(image, classes[class_id], (bounding_box[0], bounding_box[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

      # Make a decision based on the detected object
      if class_id == 0:
        # The object is a stop sign, so we need to stop the car
        print("Stop sign detected!")
      elif class_id == 1:
        # The object is a yield sign, so we need to slow down and yield to traffic
        print("Yield sign detected!")
      elif class_id == 2:
        # The object is a speed limit sign, so we need to slow down to the speed limit
        print("Speed limit sign detected!")
      elif class_id == 3:
        # The object is a turn left sign, so we need to turn left
        print("Turn left sign detected!")
      elif class_id == 4:
        # The object is a turn right sign, so we need to turn right
        print("Turn right sign detected!")

  # Return the image with the detected objects
  return image

# Capture the video feed from the camera
cap = cv2.VideoCapture(0)

while True:
  # Capture the current frame from the video feed
  frame = cap.read()[1]

  # Convert the image to a NumPy array
  image = np.array(frame)

  # Detect the traffic signs in the current frame
  image = detect_traffic_signs(image)

  # Display the frame
  cv2.imshow('Traffic Sign Detection', image)

  # Check if the user wants to quit
  key = cv2.waitKey(1) & 0xFF

  if key == 27:
    break

# Release the camera
cap.release()

# Close all windows
cv2.destroyAllWindows()
