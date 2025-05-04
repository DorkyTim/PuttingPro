import cv2
import numpy as np
import torch
import time
import math
import pathlib

# Ensure compatibility for torch hub on Windows
pathlib.PosixPath = pathlib.WindowsPath

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class BallTracker:
    def __init__(self, yolo_path='best.pt', confidence=0.3):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_path)
        self.model.conf = confidence

        self.next_ball_id = 0
        self.tracked_balls = {}  # ball_id: {"pos", "last_seen", "kalman", "radius", "prediction"}
        self.fade_duration = 1.0  # seconds
        self.dist_check = 50  # pixels to check identity

    def _create_kalman_filter(self, init_x, init_y):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.statePre = np.array([[init_x], [init_y], [0], [0]], dtype=np.float32)
        kalman.statePost = np.array([[init_x], [init_y], [0], [0]], dtype=np.float32)
        return kalman

    def detect_ball(self, frame):
        frame = cv2.resize(frame, (854, 480))
        current_time = time.time()

        for ball_id, info in self.tracked_balls.items():
            prediction = info['kalman'].predict()
            pred_pos = (int(prediction[0]), int(prediction[1]))
            self.tracked_balls[ball_id]['prediction'] = pred_pos

        results = self.model(frame)
        detections = results.pandas().xyxy[0]

        matched_ids = set()
        if len(detections) > 0:
            for _, row in detections.iterrows():
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                radius = int(max(x2 - x1, y2 - y1) / 2)

                min_dist = float('inf')
                matched_id = None
                for ball_id, info in self.tracked_balls.items():
                    pred_pos = info['prediction']
                    dist = math.hypot(center[0] - pred_pos[0], center[1] - pred_pos[1])
                    if dist < self.dist_check and dist < min_dist and current_time - info['last_seen'] < self.fade_duration:
                        min_dist = dist
                        matched_id = ball_id

                if matched_id is None:
                    matched_id = self.next_ball_id
                    self.next_ball_id += 1
                    self.tracked_balls[matched_id] = {
                        "kalman": self._create_kalman_filter(center[0], center[1]),
                        "pos": center,
                        "last_seen": current_time,
                        "radius": radius,
                        "prediction": center
                    }
                else:
                    kalman = self.tracked_balls[matched_id]['kalman']
                    prev_pos = self.tracked_balls[matched_id]['pos']
                    dt = current_time - self.tracked_balls[matched_id]['last_seen']
                    if dt > 0:
                        vx = (center[0] - prev_pos[0]) / dt
                        vy = (center[1] - prev_pos[1]) / dt
                        alpha = 0.4
                        prev_vx = kalman.statePost[2]
                        prev_vy = kalman.statePost[3]
                        vx = alpha * vx + (1 - alpha) * prev_vx
                        vy = alpha * vy + (1 - alpha) * prev_vy
                        kalman.statePost[2] = vx
                        kalman.statePost[3] = vy
                    kalman.correct(np.array([[np.float32(center[0])], [np.float32(center[1])]]))

                    self.tracked_balls[matched_id]['pos'] = center
                    self.tracked_balls[matched_id]['last_seen'] = current_time
                    self.tracked_balls[matched_id]['radius'] = radius
                    self.tracked_balls[matched_id]['prediction'] = center

                matched_ids.add(matched_id)

        # Remove expired tracks
        for ball_id in list(self.tracked_balls.keys()):
            if ball_id not in matched_ids:
                time_since_seen = current_time - self.tracked_balls[ball_id]['last_seen']
                if time_since_seen >= self.fade_duration * 2:
                    del self.tracked_balls[ball_id]

        return {
            'frame_size': (frame.shape[1], frame.shape[0]),
            'balls': {
                ball_id: {
                    'pos': info['pos'],
                    'radius': info['radius'],
                    'prediction': info['prediction'],
                    'last_seen': info['last_seen']
                }
                for ball_id, info in self.tracked_balls.items()
                if current_time - info['last_seen'] < self.fade_duration
            }
        }
