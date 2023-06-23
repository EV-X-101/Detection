import eventlet
import numpy as np
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import RPi.GPIO as GPIO
import time


# Set up GPIO pins for motor control
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# Motor 1 pins
in1 = 11
in2 = 13
en1 = 10

# Motor 2 pins
in3 = 16
in4 = 18
en2 = 15

GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(en1,GPIO.OUT)
GPIO.setup(en2,GPIO.OUT)

p1 = GPIO.PWM(en1,1000)
p2 = GPIO.PWM(en2,1000)
p1.start(0)
p2.start(0)

def img_preprocess(img):
    img = img[50:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
 
 
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
 
 
def send_control(steering_angle, throttle):
    # Convert steering angle to motor speed
    motor_speed = max(min(throttle * 100, 100), 0)
    # Set motor directions
    if steering_angle > 0:
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.HIGH)
    else:
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.HIGH)
        GPIO.output(in3,GPIO.HIGH)
        GPIO.output(in4,GPIO.LOW)
    # Set motor speeds
    p1.ChangeDutyCycle(motor_speed)
    p2.ChangeDutyCycle(motor_speed)


if __name__ == '__main__':
    model = load_model('Models/model51.h5')
    speed_limit = 30  # Set speed limit for throttle calculation
    try:
        eventlet.wsgi.server(eventlet.listen(('', 4567)), telemetry)
    except KeyboardInterrupt:
        pass
    finally:
        p1.stop()
        p2.stop()
        GPIO.cleanup()

