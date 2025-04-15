import cv2
import numpy as np
import torch
import time
import math
from collections import deque
import pathlib

# Ensure compatibility for torch hub
pathlib.PosixPath = pathlib.WindowsPath

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class BallTracker:
    def __init__(self, yolo_path='best.pt', confidence=0.3):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_path)
        self.model.conf = confidence

        self.kalmans = {}  # Dictionary to store Kalman filters by ID
        self.next_id = 1
        self.tracked_balls = {}  # id: {center, last_seen, kalman, visible, radius}

        self.prev_time = time.time()

    def create_kalman_filter(self):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        return kalman

    def assign_id_to_ball(self, center):
        min_dist = float('inf')
        matched_id = None
        for ball_id, data in self.tracked_balls.items():
            prev_center = data['center']
            if prev_center is None:
                continue
            dist = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
            if dist < 50 and dist < min_dist:
                min_dist = dist
                matched_id = ball_id

        if matched_id is not None:
            return matched_id

        new_id = self.next_id
        self.next_id += 1
        self.kalmans[new_id] = self.create_kalman_filter()
        self.tracked_balls[new_id] = {
            'center': None,
            'last_seen': 0,
            'visible': False,
            'radius': 20
        }
        return new_id

    def detect_ball(self, frame):
        frame = cv2.resize(frame, (854, 480))
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        current_time = time.time()

        visible_ids = set()

        # YOLO detection
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = int(max(x2 - x1, y2 - y1) / 2)

            ball_id = self.assign_id_to_ball(center)
            kalman = self.kalmans[ball_id]
            kalman.correct(np.array([[np.float32(center[0])], [np.float32(center[1])]]))

            self.tracked_balls[ball_id]['center'] = center
            self.tracked_balls[ball_id]['last_seen'] = current_time
            self.tracked_balls[ball_id]['visible'] = True
            self.tracked_balls[ball_id]['radius'] = radius

            cv2.circle(output, center, radius, (0, 255, 0), 2)
            cv2.putText(output, f"ID {ball_id} (YOLO)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            visible_ids.add(ball_id)

        # HoughCircles fallback for each missing tracked ball
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                   param1=100, param2=50, minRadius=5, maxRadius=100)
        if circles is not None:
            for c in np.uint16(np.around(circles[0, :])):
                x, y, r = c
                center = (int(x), int(y))
                ball_id = self.assign_id_to_ball(center)

                if ball_id in visible_ids:
                    continue

                kalman = self.kalmans[ball_id]
                kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))
                self.tracked_balls[ball_id]['center'] = center
                self.tracked_balls[ball_id]['last_seen'] = current_time
                self.tracked_balls[ball_id]['visible'] = True
                self.tracked_balls[ball_id]['radius'] = int(r)

                cv2.circle(output, center, int(r), (0, 165, 255), 2)
                cv2.putText(output, f"ID {ball_id} (Hough)", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                visible_ids.add(ball_id)

        # Prediction for not visible balls
        for ball_id, data in self.tracked_balls.items():
            if ball_id not in visible_ids:
                prediction = self.kalmans[ball_id].predict()
                x_pred = prediction[0].item()
                y_pred = prediction[1].item()
                center = (int(x_pred), int(y_pred))
                self.tracked_balls[ball_id]['center'] = center
                self.tracked_balls[ball_id]['visible'] = False

                cv2.circle(output, center, data['radius'], (0, 255, 255), 2)
                cv2.putText(output, f"ID {ball_id} (Predicted)", (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        self.prev_time = current_time

        return output
