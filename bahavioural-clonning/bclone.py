import cv2
import torch
import numpy as np
import socket

from tensorflow.keras.models import load_model
import numpy as np

import os 
import dotenv

dotenv.load_dotenv()

# load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('C:/Users/firao/Documents/Detection and Depth Estimation/Detection/bahavioural-clonning/bc.h5')


# Create a socket object
s = socket.socket()         
host = 'RPI_ADDRESS' # Get this from your raspberry pi
port = 5000 # The port used by your Raspberry Pi script

# Connect to the server
s.connect((host, port))

def predict(image):
    # Preprocess image
    image = cv2.resize(image, (200, 66))  # resize to match the size the model expects
    image = image / 255.  # normalize pixel values
    image = np.expand_dims(image, axis=0)  # add an extra dimension for batch size
    
    # Pass the image through the model
    output = model.predict(image)

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

    # As we don't have separate information for the movement, we will always move forward
    move_command = 'forward'

    return move_command, turn_command



# Main loop
cap = cv2.VideoCapture(0)  # Assuming you're using a webcam or similar device
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Our operations on the frame come here
    move, turn = predict(frame)
    
    # Send commands to the Raspberry Pi
    s.send(move.encode())
    s.send(turn.encode())
    
cap.release()
cv2.destroyAllWindows()
s.close()
