import RPi.GPIO as GPIO
import time
import socket

# Set mode and pins
GPIO.setmode(GPIO.BOARD)

# Motor 1 pins
in1 = 11
in2 = 13
en1 = 15

# Motor 2 pins
in3 = 16
in4 = 18
en2 = 22

# Set up GPIO pins
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(en1,GPIO.OUT)
GPIO.setup(en2,GPIO.OUT)

# Set up PWM
p1 = GPIO.PWM(en1,100000)
p2 = GPIO.PWM(en2,100000)
p1.start(25)
p2.start(25)

# Define motor control functions
def forward():
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.HIGH)
    GPIO.output(in4,GPIO.LOW)

def backward():
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)

def stop():
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.LOW)

def speed(duty_cycle):
    p1.ChangeDutyCycle(duty_cycle)
    p2.ChangeDutyCycle(duty_cycle)

# Set up socket connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('0.0.0.0', 5000))
sock.listen()
print('Socket connection established')
# Main program loop
while True:
    conn, addr = sock.accept()
    print(f"Connected by {addr}")
    while True:
        data = conn.recv(1024)
        if not data:
            break
        command = data.decode('utf-8')
        print(f"Received command: {command}")
        if command == 'forward':
            forward()
        elif command == 'backward':
            backward()
        elif command == 'stop':
            stop()
        elif command == 'low':
            speed(25)
        elif command == 'medium':
            speed(50)
        elif command == 'high':
            speed(75)
        elif command == 'max':
            speed(100)
        elif command == 'exit':
            stop()
            GPIO.cleanup()
            break
        else:
            print("Invalid command")
    conn.close()
