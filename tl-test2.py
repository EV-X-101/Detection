import os
import time
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from core.utils import utils as deep_utils
from core.utils import yolov5_utils
from core.config import cfg
from core.yolov5 import filter_boxes
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

def traffic_light_state(frame, bbox):
    xmin, ymin, xmax, ymax = [int(coord) for coord in bbox]
    cropped_frame = frame[ymin:ymax, xmin:xmax]

    if cropped_frame.size == 0:
        print("Empty cropped frame!")
        return "Unknown"

    # Convert the cropped image to BGR and then to HSV color space
    bgr_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
    red_lower_range1 = np.array([0, 70, 50])
    red_upper_range1 = np.array([10, 255, 255])

    red_lower_range2 = np.array([160, 70, 50])
    red_upper_range2 = np.array([180, 255, 255])

    yellow_lower_range = np.array([20, 100, 100])
    yellow_upper_range = np.array([30, 255, 255])

    green_lower_range = np.array([40, 70, 70])
    green_upper_range = np.array([90, 255, 255])

    # Create masks for each color range
    red_mask1 = cv2.inRange(hsv_frame, red_lower_range1, red_upper_range1)
    red_mask2 = cv2.inRange(hsv_frame, red_lower_range2, red_upper_range2)
    red_mask = cv2.addWeighted(red_mask1, 1.0, red_mask2, 1.0, 0.0)

    yellow_mask = cv2.inRange(hsv_frame, yellow_lower_range, yellow_upper_range)
    green_mask = cv2.inRange(hsv_frame, green_lower_range, green_upper_range)

    # Count the number of pixels for each color
    red_count = cv2.countNonZero(red_mask)
    yellow_count = cv2.countNonZero(yellow_mask)
    green_count = cv2.countNonZero(green_mask)

    # If yellow pixels are detected, prioritize yellow color
    if yellow_count > 10:
        return "Yellow"

    # Determine the traffic light state based on the number of colored pixels
    color_counts = [("Red", red_count), ("Green", green_count)]
    color_counts.sort(key=lambda x: x[1], reverse=True)
    return color_counts[0][0]

def main(save=False, info=False, count_objects=False):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'deep_sort_model/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    weights_path = 'yolov5s.pt'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(weights_path, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()

    # get video ready to save locally if flag is set
    if save:
        # by default VideoCapture returns float instead of int
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('', codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended!')
            break

        frame_num +=1

        frame_size = frame.shape[:2]
        img = cv2.resize(frame, (640, 640))
        img = torch.from_numpy(img).to(device)
        img = img.permute(2, 0, 1).float().unsqueeze(0) / 255.0

        start_time = time.time()

        # run detection
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)

        # convert detections to deep sort format
        detections = []
        for i, det in enumerate(pred):
            if len(det):
                det = det[:, :4]
                det[:, 0] *= frame_size[1] / 640
                det[:, 1] *= frame_size[0] / 640
                det[:, 2] *= frame_size[1] / 640
                det[:, 3] *= frame_size[0] / 640
                conf = det[:, -1]
                if len(det) > 1:
                    det = np.array([det[np.argmax(conf)]])
                for *xyxy, conf in det:
                    detections.append(Detection(xyxy, conf, "person", encoder(frame, xyxy)))

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            # Check if the detected object is a traffic light and get its state
            if class_name == "traffic light":
                light_state = traffic_light_state(frame, bbox)
                class_name = f"{class_name} ({light_state})"
                if light_state == "Red":
                    color = (255,0,0)
                elif light_state =="Yellow":
                    color = (181, 176, 20)
                elif light_state == "Green":
                    color = (35, 120, 18)

            # draw bbox on screen
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # if enable info flag then print details about each track
            if info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if save:
            out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cv2.destroyAllWindows()
