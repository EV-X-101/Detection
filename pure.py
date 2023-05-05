import cv2
import torch
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Load the pre-trained YOLOv5 model
model = attempt_load("yolov5s.pt", map_location=torch.device('cpu'))

# Set the input size of the model (the default size is 640x640)
input_size = (640, 640)

# Set the output classes of the model (there are 80 classes in the COCO dataset)
output_classes = [9]  # Traffic Light class id is 9

# Set the detection threshold
threshold = 0.5

def detect_traffic_lights(image):
    # Resize the image to the input size of the model
    img = cv2.resize(image, input_size)

    # Convert the image to a PyTorch tensor
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

    # Select the device for inference
    device = select_device('')

    # Pass the image through the model to get the detections
    with torch.no_grad():
        output = model(img.to(device))[0]

    # Apply non-maximum suppression to remove overlapping detections
    output = non_max_suppression(output, conf_thres=threshold, classes=output_classes)[0]

    # Loop through the detections and extract the bounding boxes and confidence scores
    bboxes = []
    scores = []
    for det in output:
        x1, y1, x2, y2, conf, cls = det.tolist()

        # Rescale the bounding box coordinates to the original image size
        bbox = scale_coords(img.shape[2:], det[:4], image.shape[:2]).tolist()
        bbox = [int(x) for x in bbox]

        if cls == 9:  # Traffic Light class id is 9
            bboxes.append(bbox)
            scores.append(conf)

    return bboxes, scores

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Detect traffic lights in the frame
    bboxes, scores = detect_traffic_lights(frame)

    # Draw bounding boxes and confidence scores on the frame
    for bbox, score in zip(bboxes, scores):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Traffic Light ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the image with the traffic light detection results
    cv2.imshow("Traffic Light Detection", frame)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
