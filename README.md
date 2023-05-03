# Object Detection and Depth Estimation 🔎📏

This repository contains code for performing object detection and depth estimation using Python, PyTorch, and YOLOv5. The code uses a pre-trained YOLOv5 model to detect objects in a live video stream from a camera and estimates the distance to the detected objects using stereo vision.

## Prerequisites 🧑‍💻

To use this code, you'll need to have the following software installed:

- Anaconda: https://www.anaconda.com/products/individual 🐍
- PyCharm or VS Code: https://www.jetbrains.com/pycharm/download/ or https://code.visualstudio.com/download 💻

## Installation 🚀

To get started, follow these steps:

1. Install Anaconda by downloading and running the installer for your operating system from the Anaconda website.

2. Open PyCharm or VS Code and create a new project in a directory of your choice.

3. Open a terminal window within PyCharm or VS Code and create a new conda environment by running the following command:

`conda create --name detection_env`

4. Activate the new conda environment by running the following command:

`conda activate detection_env`


5. Install PyTorch with GPU support by running the following command:

`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

N.B: To check weather torch is using your GPU or not, you can use the following code.

```
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the current CUDA device index and its name
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"CUDA is available. Using GPU: {device_name} (device {current_device})")
else:
    print("CUDA is not available. Using CPU.")
```

This command installs PyTorch version 2.0 with CUDA 11.8 support.

6. Clone the YOLOv5 repository and install the remaining dependencies by running the following commands:

```
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

The first command clones the YOLOv5 repository, and the second and third commands install the remaining dependencies.

7. You're now ready to run the code! To run the script, execute the following command:

`python depth-estimation.py`


The script will open a window showing the live video stream from the camera with the detected objects and estimated distances overlaid on the video.

To exit the script, press the 'q' key.

## Example Output 📷

Here are some example output images from the script:

- Object detection: ![Object Detection Example](images/depth1.png) 🕵️‍♀️
- Depth estimation: ![Depth Estimation Example](images/detection1.png) 📏


# 🚀 RC Car Object Detection and Obstacle Avoidance

This project demonstrates how to use object detection and stereo vision to enable obstacle avoidance on an RC car. The car is equipped with a camera and two motors controlled by a Raspberry Pi. The camera captures images that are analyzed by a deep learning model (YOLOv5) running on a PC. The model detects objects in the scene and estimates their distance using stereo vision. If a potential obstacle (e.g., person, car, motorcycle, bicycle) is detected, a command is sent to the Raspberry Pi to stop the car.

## 📝 Requirements

- Python 3
- PyTorch (>=1.7.0)
- OpenCV-Python (>=4.4.0)
- Raspberry Pi with Raspbian OS
- Two DC motors with wheels and a motor driver (e.g., L298N)
- Camera (e.g., USB webcam)
- Jumper wires and breadboard

## 🛠️ Hardware Setup

1. Connect the motors to the motor driver (L298N) following the wiring diagram in the `motor_control` folder. Connect the motor driver to the Raspberry Pi using jumper wires.

2. Connect the camera to the Raspberry Pi using a USB cable.

3. Place the motors and the camera on the RC car chassis and connect them to the power source (e.g., batteries).

## 📊 Software Setup

1. Clone this repository to your PC and Raspberry Pi.

2. Install the required libraries listed above using pip.

3. Upload the `motor_control.py` script to the Raspberry Pi and run it.

4. Modify the IP address in the `object_detection.py` script to match the IP address of your Raspberry Pi. Upload the modified script to your PC and run it.

5. Place the RC car on a flat surface with no obstacles and start the `object_detection.py` script on your PC. The car should move forward.

6. Test the obstacle avoidance by placing a potential obstacle (e.g., person, car, motorcycle, bicycle) in front of the car. The car should stop.

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🤝 Acknowledgments

- YOLOv5: https://github.com/ultralytics/yolov5
- PyTorch: https://pytorch.org/
- OpenCV: https://opencv.org/
- Raspberry Pi: https://www.raspberrypi.org/
- Adafruit: https://learn.adafruit.com/


## Credits 🙏

This code is based on the YOLOv5 object detection tutorial by Ultralytics: https://github.com/ultralytics/yolov5

## Contributing 🛠️

Contributions to this project are welcome! If you find a bug or would like to add a new feature, please open an issue or submit a pull request.

## License 📃

This project is licensed under the MIT License. You may use, distribute, and modify this code as long as you include the original copyright notice and license terms. See LICENSE.txt for details.



