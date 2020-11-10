#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from collections import deque
from keras import backend
from multiprocessing import Process, Queue
import time


def receive_data(queue):
    print('Process(%s) is receiving...' % os.getpid())
    rtsp = "rtsp://admin:bocom123456@10.20.40.205:554"
    cap = cv2.VideoCapture(rtsp)
    print(cap.isOpened())
    while cap.isOpened():
        ret, img_ori = cap.read()
        if ret:
            if queue.full():
                queue.get()
            else:
                queue.put(img_ori)
        else:
            cap.release()
            # read again
            cap = cv2.VideoCapture(rtsp)


def process_data(queue):
    print('Process(%s) is process...' % os.getpid())
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, 15, (1280, 720))
    while True:
        frame = queue.get(True)
        # process the image, add your business logic
        t1 = time.time()
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, class_names = yolo.detect_image(image)
        features = encoder(frame, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150, (color), 2)
            if len(class_names) > 0:
                cv2.putText(frame, str(class_names[0]), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, (color),
                            2)

            i += 1
            # bbox_center_point(x,y)
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            # track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            # center point
            cv2.circle(frame, (center), 1, color, thickness)

            # draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue

        count = len(set(counter))
        global fps
        fps = (fps + (1. / (time.time() - t1))) / 2
        cv2.putText(frame, "Total Object Counter: " + str(count), (int(20), int(120)), 0, 5e-3 * 200, (0, 255, 0),
                    2)
        cv2.putText(frame, "Current Object Counter: " + str(i), (int(20), int(80)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "FPS: %f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
        # cv2.putText(frame, "time(s): %f" % time_use, (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
        cv2.namedWindow("YOLO3_Deep_SORT", 0)
        cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768)
        cv2.imshow('YOLO3_Deep_SORT', frame)
        videoWriter.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # cap.release()
    cv2.destroyAllWindows()
    yolo.close_session()


if __name__ == '__main__':
    backend.clear_session()
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", help="path to input video", default="./test_video/test.avi")
    ap.add_argument("-c", "--class", help="name of class", default="person")
    args = vars(ap.parse_args())

    pts = [deque(maxlen=5) for _ in range(9999)]
    warnings.filterwarnings('ignore')

    # initialize a list of colors to represent each possible class label
    np.random.seed(100)
    COLORS = np.random.randint(0, 255, size=(200, 3),
                               dtype="uint8")

    # Definition of the parameters
    max_cosine_distance = 0.5  # 余弦距离的控制阈值
    nn_budget = None
    nms_max_overlap = 0.3  # 非极大抑制的阈值

    counter = []
    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    yolo = YOLO()

    fps = 0.0
    q = Queue(maxsize=2)
    receive_data_no = Process(target=receive_data, args=(q,))
    receive_data_no.start()
    # receive_data_no.join()
    process_data(q)
